"""v2 audio->motion demo: generate motion from a BEAT2 test clip with the trained v2 model
(FAT5 + GLM-4-Voice audio + ViBES 1x face + ViBES root translation), render it, mux the audio.

  python demo_v2_a2m.py --cfg configs/config_mixed_stage3_a2m_v2.yaml --nodebug
"""
import os, glob, subprocess
import numpy as np
import torch
import torchaudio
import pytorch_lightning as pl
import smplx
from lom.config import parse_args
from lom.data.build_data import build_data
from lom.models.build_model import build_model
from lom.utils.load_checkpoint import load_pretrained_lm, load_pretrained_vae_compositional
from lom.utils.logger import create_logger
from lom.utils.rotation_conversions import rotation_6d_to_axis_angle

CKPT = 'experiments/lom/Instruct_Mixed_A2M_v2_2026-06-22-17-20-27/checkpoints/last.ckpt'
OUT = 'experiments/v2_vis_a2m'
SMPLX_DIR = 'model_files/smplx_models/'


def render_and_mux(rec_pose, rec_exps, rec_trans, rec_beta, raw_audio, tag):
    os.makedirs(OUT, exist_ok=True)
    dev = 'cpu'; n = rec_pose.shape[0]
    sm = smplx.create(SMPLX_DIR, model_type='smplx', gender='NEUTRAL_2020', use_face_contour=False,
                      num_betas=300, num_expression_coeffs=100, ext='npz', use_pca=False).eval().to(dev)
    rp = rec_pose.float().to(dev)
    verts = sm(betas=rec_beta.float().reshape(1, 300).tile(n, 1).to(dev),
               transl=rec_trans.float().reshape(n, 3).to(dev),
               expression=rec_exps.float().reshape(n, 100).to(dev),
               jaw_pose=rp[:, 66:69], global_orient=rp[:, :3], body_pose=rp[:, 3:66],
               left_hand_pose=rp[:, 75:120], right_hand_pose=rp[:, 120:165],
               leye_pose=rp[:, 69:72], reye_pose=rp[:, 72:75]).vertices.detach().cpu().numpy()
    np.save(os.path.join(OUT, f'{tag}.npy'), verts)
    print(f"[render] {verts.shape}; blender...", flush=True)
    subprocess.run(['./third_party/blender-2.93.18-linux-x64/blender', '--background', '--python', 'render.py',
                    '--', '--cfg=./configs/render.yaml', f'--dir={OUT}', '--mode=video'], capture_output=True, text=True)
    vids = sorted(glob.glob(f'{OUT}/*.mp4'), key=os.path.getmtime)
    if not vids:
        print("[render] no video"); return None
    vid = vids[-1]
    # save raw_audio (16 kHz) and mux
    wav = os.path.join(OUT, f'{tag}.wav')
    aud = raw_audio.detach().float().cpu().reshape(1, -1)
    torchaudio.save(wav, aud, 16000)
    out_av = os.path.join(OUT, f'{tag}_audio.mp4')
    r = subprocess.run(['ffmpeg', '-y', '-i', vid, '-i', wav, '-c:v', 'copy', '-c:a', 'aac', '-shortest', out_av],
                       capture_output=True, text=True)
    print(f"[mux] rc={r.returncode} -> {out_av}", flush=True)
    return out_av if r.returncode == 0 else vid


def main():
    cfg = parse_args(phase="test")
    cfg.TEST.CHECKPOINTS = CKPT
    cfg.TEST.CHECKPOINTS_FACE = './model_files/pretrained_cpt/face/face_lom.ckpt'
    cfg.TEST.CHECKPOINTS_HAND = './model_files/pretrained_cpt/lom_vq_ds/lom_vq.ckpt'
    cfg.TEST.CHECKPOINTS_UPPER = './model_files/pretrained_cpt/lom_vq_ds/lom_vq.ckpt'
    cfg.TEST.CHECKPOINTS_LOWER = './model_files/pretrained_cpt/lom_vq_ds/lom_vq.ckpt'
    cfg.TEST.CHECKPOINTS_GLOBAL = './model_files/pretrained_cpt/VAE_Global_from_Lower54/vibes_global.ckpt'
    cfg.TEST.BATCH_SIZE = 1
    logger = create_logger(cfg, phase="test")
    pl.seed_everything(cfg.SEED_VALUE)

    model = build_model(cfg)
    load_pretrained_vae_compositional(cfg, model, logger, phase="test")
    load_pretrained_lm(cfg, model, logger, phase="test")
    model = model.cuda().eval()

    dm = build_data(cfg)
    batch = next(iter(dm.test_dataloader()))
    for k in list(batch.keys()):
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].cuda()

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        rs = model.val_a2m_forward(batch)
    rec_pose_6d = rs["rec_pose"].float()             # (bs, n, j*6)
    bs, n, d = rec_pose_6d.shape; j = d // 6
    rec_pose = rotation_6d_to_axis_angle(rec_pose_6d.reshape(bs * n, j, 6)).reshape(bs * n, j * 3)[:n]
    rec_trans = rs["rec_trans"].float().reshape(-1, 3)[:n]
    rec_exps = rs["rec_exps"].float().reshape(-1, 100)[:n]
    rec_beta = rs["tar_beta"].float().reshape(-1, 300)[0]
    ra = rs["raw_audio"]
    raw_audio = torch.as_tensor(np.asarray(ra[0] if isinstance(ra, (list, tuple)) else ra)).float()
    print("SHAPES pose", tuple(rec_pose.shape), "trans", tuple(rec_trans.shape),
          "exps", tuple(rec_exps.shape), "audio", tuple(raw_audio.shape), flush=True)
    pf = float((rec_pose[1:] - rec_pose[:-1]).abs().mean())
    print(f"PER-FRAME pose change: {pf:.6f}  (greedy was ~0; higher = moving)", flush=True)
    L = min(rec_pose.shape[0], 375)   # cap render for faster feedback
    print("v2 a2m ->", render_and_mux(rec_pose[:L], rec_exps[:L], rec_trans[:L], rec_beta, raw_audio, 'v2_a2m'), flush=True)


if __name__ == "__main__":
    main()
