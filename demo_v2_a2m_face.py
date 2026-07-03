"""v2 audio->face demo: generate a co-speech FLAME face from a BEAT2/TFHP test clip with the
trained v2 model, then render it with LoM's ViBES-derived PyTorch3D renderer (HardPhongShader).

Two stages, possibly in two envs (the LoM model env ships no pytorch3d/av):
  1. GENERATE (LoM model env): run the model -> 112-D face feature `rec_face`; cache it + the audio.
  2. RENDER   (pytorch3d env): FLAME -> RenderMesh -> mp4. Done in-process if pytorch3d is importable,
     otherwise run `scripts/render_face_flame.py` on the cached _rec_face.npy in e.g. the ViBES env.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    python demo_v2_a2m_face.py --cfg configs/config_mixed_stage3_a2m_v2.yaml --nodebug
"""
import os
import numpy as np
import torch
import pytorch_lightning as pl
from lom.config import parse_args
from lom.data.build_data import build_data
from lom.models.build_model import build_model
from lom.utils.load_checkpoint import load_pretrained_lm, load_pretrained_vae_compositional
from lom.utils.logger import create_logger

CKPT = 'experiments/lom/Instruct_Mixed_A2M_v2/checkpoints/last.ckpt'   # edit to your trained checkpoint
FLAME_DIR = 'model_files/FLAME2020'
OUT = 'experiments/v2_vis_a2m_face'
MAXF = 375          # cap rendered frames (375 @ 25fps = 15s)
SAMPLE_RATE = 16000


def generate_face(cfg):
    """Run the model on one test clip -> cache (rec_face, raw_audio)."""
    cfg.TEST.CHECKPOINTS = CKPT
    cfg.TEST.CHECKPOINTS_FACE = './model_files/pretrained_cpt/face/face_lom.ckpt'
    cfg.TEST.CHECKPOINTS_HAND = './model_files/pretrained_cpt/lom_vq_ds/lom_vq.ckpt'
    cfg.TEST.CHECKPOINTS_UPPER = './model_files/pretrained_cpt/lom_vq_ds/lom_vq.ckpt'
    cfg.TEST.CHECKPOINTS_LOWER = './model_files/pretrained_cpt/lom_vq_ds/lom_vq.ckpt'
    cfg.TEST.CHECKPOINTS_GLOBAL = './model_files/pretrained_cpt/VAE_Global_from_Lower54/vibes_global.ckpt'
    cfg.TEST.BATCH_SIZE = 1
    logger = create_logger(cfg, phase='test')
    pl.seed_everything(cfg.SEED_VALUE)
    model = build_model(cfg)
    load_pretrained_vae_compositional(cfg, model, logger, phase='test')
    load_pretrained_lm(cfg, model, logger, phase='test')
    model = model.cuda().eval()
    dm = build_data(cfg)
    batch = next(iter(dm.test_dataloader()))
    for k in list(batch.keys()):
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].cuda()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        rs = model.val_a2m_forward(batch)
    rec_face = rs['rec_face'].float()[0].cpu().numpy()
    ra = rs['raw_audio']
    raw_audio = np.asarray(ra[0] if isinstance(ra, (list, tuple)) else ra, dtype=np.float32).reshape(-1)
    return rec_face, raw_audio


def main():
    os.makedirs(OUT, exist_ok=True)
    cf, ca = os.path.join(OUT, '_rec_face.npy'), os.path.join(OUT, '_audio.npy')
    if os.path.exists(cf):
        print('using cached rec_face/audio (re-render only)', flush=True)
        rec_face, raw_audio = np.load(cf), np.load(ca)
    else:
        cfg = parse_args(phase='test')
        rec_face, raw_audio = generate_face(cfg)
        np.save(cf, rec_face)
        np.save(ca, raw_audio)
        print(f'cached rec_face {rec_face.shape} -> {cf}', flush=True)

    try:
        from lom.render.flame_render import render_face_sequence, _HAS_PYTORCH3D
        if not _HAS_PYTORCH3D:
            raise ImportError('pytorch3d not available')
        out_path = os.path.join(OUT, 'face.mp4')
        n = render_face_sequence(rec_face, FLAME_DIR, out_path, audio=raw_audio,
                                 fps=25, sample_rate=SAMPLE_RATE, max_frames=MAXF)
        print(f'FACE VIDEO -> {out_path}  ({n} frames)', flush=True)
    except ImportError:
        print(f'\n[render skipped] pytorch3d/av not in this env. Cached the face feature instead.\n'
              f'Render in a pytorch3d env (e.g. the ViBES conda env) with:\n'
              f'  PYTHONPATH=. python scripts/render_face_flame.py \\\n'
              f'      --rec_face {cf} --audio {ca} --out {os.path.join(OUT, "face.mp4")} '
              f'--flame_dir {FLAME_DIR}\n', flush=True)


if __name__ == '__main__':
    main()
