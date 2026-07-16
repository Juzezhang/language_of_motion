"""4-part (face+upper+lower+hand) inference for the decoder-only a2m model.

Generate 4 self-delimiting blocks (length-enforced: body L = audio/2, face = 4L) with each
block restricted to its codebook range -> decode the 4 VQs -> reconstruct SMPL-X pose via the
exact val_a2m_forward math -> SMPL-X vertices -> Blender render + audio mux.

Run with PYTHONPATH=<repo>, HF_HOME set, lom_release python.
"""
import os, sys, glob, argparse, subprocess
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fourpart as FP
from transformers import LogitsProcessor, LogitsProcessorList
from lom.archs.lom_vq import (VQVAEConvZeroDSUS1_PaperVersion, VQVAEConvZeroDSUS_PaperVersion, VAEConvZero)
from lom.data.mixed_dataset.data_tools import JOINT_MASK_UPPER, JOINT_MASK_LOWER, JOINT_MASK_HAND
from lom.utils.rotation_conversions import (rotation_6d_to_axis_angle, rotation_6d_to_matrix,
                                            matrix_to_axis_angle, matrix_to_rotation_6d)
from lom.utils.other_tools import integrate_local_velocity

FACE_VQ = './model_files/pretrained_cpt/face/face_lom.ckpt'
BODY_VQ = './model_files/pretrained_cpt/lom_vq_ds/lom_vq.ckpt'
GLOBAL_VQ = './model_files/pretrained_cpt/VAE_Global_from_Lower54/vibes_global.ckpt'
SMPLX_DIR = 'model_files/smplx_models/'


def _load_sub(vae, ckpt, prefix):
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    miss, unexp = vae.load_state_dict(sub, strict=False)
    print(f"[vq] {prefix} loaded (miss={len(miss)} unexp={len(unexp)})", flush=True)
    return vae


def load_vqs(device):
    vf = _load_sub(VQVAEConvZeroDSUS1_PaperVersion(vae_layer=3, code_num=512, codebook_size=512, vae_quantizer_lambda=1, vae_test_dim=112), FACE_VQ, 'vae_face.')
    vu = _load_sub(VQVAEConvZeroDSUS_PaperVersion(vae_layer=3, code_num=256, codebook_size=256, vae_quantizer_lambda=1, vae_test_dim=78), BODY_VQ, 'vae_upper.')
    vl = _load_sub(VQVAEConvZeroDSUS_PaperVersion(vae_layer=3, code_num=256, codebook_size=256, vae_quantizer_lambda=1, vae_test_dim=54), BODY_VQ, 'vae_lower.')
    # hand prefix is vae_hands. in lom_vq.ckpt
    vh = VQVAEConvZeroDSUS_PaperVersion(vae_layer=3, code_num=256, codebook_size=256, vae_quantizer_lambda=1, vae_test_dim=180)
    sd = torch.load(BODY_VQ, map_location='cpu', weights_only=False); sd = sd.get('state_dict', sd)
    pref = 'vae_hands.' if any(k.startswith('vae_hands.') for k in sd) else 'vae_hand.'
    vh = _load_sub(vh, BODY_VQ, pref)
    vg = _load_sub(VAEConvZero(vae_layer=4, code_num=256, codebook_size=256, vae_quantizer_lambda=1, vae_test_dim=61), GLOBAL_VQ, 'vae_global.')
    return [x.float().to(device).eval() for x in (vf, vu, vl, vh, vg)]


def load_decoder(ckpt, device):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    vocab = ck['vocab']
    model, _ = FP.build_pretrained_4part(ck['cfg']['pretrained'])
    model.load_state_dict(ck['model'])
    model.to(device).eval()
    print(f"[infer] {ckpt} step={ck.get('step')} val={ck.get('val_loss')}", flush=True)
    return model, vocab


class RangeProc(LogitsProcessor):
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def __call__(self, input_ids, scores):
        m = torch.full_like(scores, float('-inf')); m[:, self.lo:self.hi] = 0.0
        return scores + m


@torch.no_grad()
def _gen_block(model, seq, n, lo, hi, sample, temp):
    gk = dict(max_new_tokens=n, min_new_tokens=n, pad_token_id=0,
              logits_processor=LogitsProcessorList([RangeProc(lo, hi)]), use_cache=True)
    gk.update(do_sample=True, temperature=temp) if sample else gk.update(do_sample=False, num_beams=1)
    attn = torch.ones_like(seq)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model.generate(input_ids=seq, attention_mask=attn, **gk)
    return out


@torch.no_grad()
def generate_4part(model, vocab, audio_tok, device, sample=True, temp=1.0, max_audio=512):
    v = vocab
    audio_tok = audio_tok[:max_audio]
    Ta = int(audio_tok.shape[0])
    L = max(1, Ta // 2)                       # body tokens = audio/2
    Lf = 4 * L                                 # face = 4x body
    au = np.clip(audio_tok, 0, FP.N_AUDIO - 1) + v['audio_offset']
    seq = np.concatenate([[v['bos']], au, [v['sep']], [v['face_s']]]).astype(np.int64)
    seq = torch.from_numpy(seq).unsqueeze(0).to(device)
    seq = _gen_block(model, seq, Lf, v['face_offset'], v['face_offset'] + FP.N_FACE, sample, temp)
    for start, off in [(v['up_s'], v['upper_offset']), (v['low_s'], v['lower_offset']), (v['hand_s'], v['hand_offset'])]:
        seq = torch.cat([seq, torch.tensor([[start]], device=device)], dim=1)
        seq = _gen_block(model, seq, L, off, off + FP.N_BODY, sample, temp)
    ids = seq[0].tolist()
    # parse blocks by delimiter positions
    def block(after_tok, off, ln):
        i = ids.index(after_tok) + 1
        return torch.tensor([c - off for c in ids[i:i + ln]], device=device).clamp(0, (FP.N_FACE if off == v['face_offset'] else FP.N_BODY) - 1).long()
    face = block(v['face_s'], v['face_offset'], Lf)
    upper = block(v['up_s'], v['upper_offset'], L)
    lower = block(v['low_s'], v['lower_offset'], L)
    hand = block(v['hand_s'], v['hand_offset'], L)
    return face, upper, lower, hand


def _inv_sel(filtered, mask, n, device):
    mask = torch.from_numpy(mask).to(device)
    out = torch.zeros((n, 165), device=device)
    sel = torch.where(mask == 1)[0]
    out[:, sel] = filtered
    return out


@torch.no_grad()
def decode_pose(vqs, face, upper, lower, hand, device):
    vae_face, vae_upper, vae_lower, vae_hand, vae_global = vqs
    rec_face = vae_face.decode(face.unsqueeze(0).int()).float()
    rec_upper = vae_upper.decode(upper.unsqueeze(0).int()).float()
    rec_lower = vae_lower.decode(lower.unsqueeze(0).int()).float()
    rec_hands = vae_hand.decode(hand.unsqueeze(0).int()).float()
    nmin = min(rec_face.shape[1], rec_upper.shape[1], rec_lower.shape[1], rec_hands.shape[1])
    rec_face, rec_upper, rec_lower, rec_hands = rec_face[:, :nmin], rec_upper[:, :nmin], rec_lower[:, :nmin], rec_hands[:, :nmin]
    rec_pose_jaw = rec_face[:, :, 6:12]; rec_exps = rec_face[:, :, 12:]
    rec_pose_legs = rec_lower[:, :, :54]
    bs, n = 1, nmin
    ru = rotation_6d_to_axis_angle(rec_upper.reshape(bs, n, 13, 6)).reshape(bs * n, 39)
    ru = _inv_sel(ru, JOINT_MASK_UPPER, bs * n, device)
    rl_m = rotation_6d_to_matrix(rec_pose_legs.reshape(bs, n, 9, 6))
    rec_lower2global = matrix_to_rotation_6d(rl_m.clone()).reshape(bs, n, 54)
    rl = matrix_to_axis_angle(rl_m).reshape(bs * n, 27); rl = _inv_sel(rl, JOINT_MASK_LOWER, bs * n, device)
    rh = matrix_to_axis_angle(rotation_6d_to_matrix(rec_hands.reshape(bs, n, 30, 6))).reshape(bs * n, 90)
    rh = _inv_sel(rh, JOINT_MASK_HAND, bs * n, device)
    jaw = matrix_to_axis_angle(rotation_6d_to_matrix(rec_pose_jaw.reshape(bs * n, 6))).reshape(bs * n, 3)
    rec_pose = ru + rl + rh
    rec_pose[:, 66:69] = jaw
    to_global = torch.nn.functional.pad(rec_lower, (0, 7)) if rec_lower.shape[2] == 54 else rec_lower
    to_global[:, :, 54:57] = 0.0; to_global[:, :, :54] = rec_lower2global
    rec_global = vae_global(to_global)
    vsel = rec_global["rec_pose"][:, :, 54:57]
    go = rec_pose[:, :3].reshape(bs, n, 3)
    rec_trans = torch.stack([integrate_local_velocity(vsel[b], go[b]) for b in range(bs)], dim=0)
    return rec_pose.reshape(n, 165), rec_exps.reshape(n, 100), rec_trans.reshape(n, 3), rec_face.reshape(n, 112)


def save_clip(out_dir, tag, rec_pose, rec_exps, rec_trans, raw_audio, fps=25, rec_face=None):
    """Compute SMPL-X verts, save verts npy + a wav trimmed to the render length + the 112-D face feat. No Blender/GPU."""
    import smplx, soundfile as sfw
    sub = os.path.join(out_dir, tag)
    os.makedirs(sub, exist_ok=True)
    for f in glob.glob(f'{sub}/*'):
        (os.remove(f) if os.path.isfile(f) else __import__('shutil').rmtree(f))
    n = rec_pose.shape[0]
    if rec_face is not None:
        np.save(os.path.join(sub, f'{tag}_face.npy'), np.asarray(rec_face.float().cpu() if hasattr(rec_face, 'cpu') else rec_face))
    sm = smplx.create(SMPLX_DIR, model_type='smplx', gender='NEUTRAL_2020', use_face_contour=False,
                      num_betas=300, num_expression_coeffs=100, ext='npz', use_pca=False).eval()
    rp = rec_pose.float().cpu()
    verts = sm(betas=torch.zeros(n, 300), transl=rec_trans.float().cpu().reshape(n, 3),
               expression=rec_exps.float().cpu().reshape(n, 100),
               jaw_pose=rp[:, 66:69], global_orient=rp[:, :3], body_pose=rp[:, 3:66],
               left_hand_pose=rp[:, 75:120], right_hand_pose=rp[:, 120:165],
               leye_pose=rp[:, 69:72], reye_pose=rp[:, 72:75]).vertices.detach().cpu().numpy()
    np.save(os.path.join(sub, f'{tag}.npy'), verts)
    if raw_audio is not None:
        sfw.write(os.path.join(sub, f'{tag}.wav'),
                  np.asarray(raw_audio)[:int(n / fps * 16000)].astype(np.float32), 16000)
    print(f"[save] {tag}: verts {verts.shape} + wav -> {sub}", flush=True)
    return sub


def blender_and_mux(sub, tag, render_cfg='./configs/render_cpu.yaml', fps=25):
    """Blender-render the verts npy in `sub` -> frames -> ffmpeg mp4@fps -> mux the wav (robust: ffmpeg, not the torchvision video writer)."""
    subprocess.run(['./third_party/blender-2.93.18-linux-x64/blender', '--background', '--python', 'render.py',
                    '--', f'--cfg={render_cfg}', f'--dir={sub}', '--mode=video'], capture_output=True, text=True)
    fr = sorted(glob.glob(f'{sub}/*_frames'))
    silent = os.path.join(sub, f'{tag}_silent.mp4')
    if fr and glob.glob(f'{fr[0]}/frame_*.png'):            # preferred: ffmpeg the frames at the right fps
        subprocess.run(['ffmpeg', '-y', '-framerate', str(fps), '-i', f'{fr[0]}/frame_%04d.png',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-loglevel', 'error', silent], text=True)
    else:                                                    # fallback: render.py's own mp4
        mp4s = [m for m in glob.glob(f'{sub}/*.mp4') if 'audio' not in m and 'silent' not in m]
        if not mp4s:
            print(f"[render] {tag}: NO VIDEO"); return None
        silent = sorted(mp4s, key=os.path.getmtime)[-1]
    wav = os.path.join(sub, f'{tag}.wav')
    if os.path.exists(wav):
        av = os.path.join(sub, f'{tag}_audio.mp4')
        r = subprocess.run(['ffmpeg', '-y', '-i', silent, '-i', wav, '-c:v', 'copy', '-c:a', 'aac',
                            '-shortest', '-loglevel', 'error', av], text=True)
        if r.returncode == 0:
            print(f"[render] {tag} -> {av}", flush=True); return av
    print(f"[render] {tag} -> {silent} (no audio)", flush=True); return silent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--n_clips', type=int, default=2)
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max_frames', type=int, default=375)
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--save_only', action='store_true')   # save verts+wav, skip Blender (render on CPU separately)
    ap.add_argument('--render_dir', default='')           # CPU: skip gen, just Blender+mux the verts in this dir
    ap.add_argument('--tfhp', action='store_true')        # drive from TFHP talking-head clips (expressive face) instead of BEAT2
    ap.add_argument('--beat2_names', default='')          # comma-separated BEAT2 clip basenames (e.g. held-out test); overrides first-N-sorted
    ap.add_argument('--audio_tokens', default='')         # EXTERNAL entry: comma-separated GLM audio-token .npy files (needs no internal data)
    ap.add_argument('--audio_wav', default='')            #   optional matching .wav(s), same order, for A/V; else the clip is silent
    args = ap.parse_args()
    if args.render_dir:                                    # CPU render step (no model / no GPU)
        tag = os.path.basename(args.render_dir.rstrip('/'))
        blender_and_mux(args.render_dir, tag, fps=args.fps)
        return
    device = 'cuda'
    model, vocab = load_decoder(args.ckpt, device)
    vqs = load_vqs(device)

    def _wav(p):
        w, _ = sf.read(p); return (w.mean(1) if w.ndim > 1 else w).astype(np.float32)

    clips = []   # (name, audio_tokens, raw_wav)
    if args.audio_tokens:                                    # EXTERNAL: user-provided GLM audio-token .npy (needs no training data)
        wavs = [w.strip() for w in args.audio_wav.split(',')] if args.audio_wav else []
        for i, tk in enumerate([t.strip() for t in args.audio_tokens.split(',') if t.strip()]):
            at = np.load(tk).astype(np.int64).reshape(-1)
            aw = _wav(wavs[i]) if i < len(wavs) and os.path.exists(wavs[i]) else None
            clips.append((os.path.splitext(os.path.basename(tk))[0], at, aw))
    elif args.tfhp:                                          # expressive talking-head driving audio (internal demo)
        cfg = FP.load_cfg()
        T = cfg.DATASET['TFHP'].ROOT
        for take in ['TH_00005/000', 'TH_00010/000'][:args.n_clips]:
            cs = sorted(glob.glob(f'{T}/audios_token_glm/{take}/*.npy'))[:6]
            if not cs:
                continue
            at = np.concatenate([np.load(c).astype(np.int64).reshape(-1) for c in cs])
            aw = np.concatenate([_wav(f'{T}/audios/{take}/' + os.path.basename(c).replace('.npy', '.wav')) for c in cs])
            clips.append((f"tfhp_{take.replace('/', '_')}", at, aw))
    else:                                                     # BEAT2 (internal demo)
        cfg = FP.load_cfg()
        B = cfg.DATASET['BEAT2'].ROOT
        if args.beat2_names:                                  # explicit clips (held-out test)
            files = [f'{B}/audios_token_glm/{n.strip()}.npy' for n in args.beat2_names.split(',') if n.strip()]
        else:
            files = sorted(glob.glob(f'{B}/audios_token_glm/*.npy'))[:args.n_clips]
        for ap_ in files:
            name = os.path.basename(ap_).replace('.npy', '')
            at = np.load(ap_).astype(np.int64).reshape(-1)
            wp = f'{B}/wave16k/{name}.wav'
            clips.append((name, at, _wav(wp) if os.path.exists(wp) else None))

    for name, at, ra in clips:
        face, upper, lower, hand = generate_4part(model, vocab, at, device,
                                                  sample=not args.greedy, temp=args.temperature)
        rec_pose, rec_exps, rec_trans, rec_face = decode_pose(vqs, face, upper, lower, hand, device)
        n = min(rec_pose.shape[0], args.max_frames)
        pf = float((rec_pose[1:n] - rec_pose[:n - 1]).abs().mean())
        print(f"[{name}] audio={at.shape[0]} -> pose {tuple(rec_pose.shape)} render {n}f  pf {pf:.5f}", flush=True)
        sub = save_clip(args.out, f'body_{name}', rec_pose[:n], rec_exps[:n], rec_trans[:n], ra, fps=args.fps, rec_face=rec_face[:n])
        if not args.save_only:
            blender_and_mux(sub, f'body_{name}', fps=args.fps)


if __name__ == '__main__':
    main()
