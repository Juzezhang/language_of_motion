"""Render a FLAME face-feature sequence (112-D) to an mp4 with LoM's ViBES-derived PyTorch3D renderer.

Runs in any env that has pytorch3d + av + smplx (e.g. the ViBES conda env), independent of LoM's
model dependencies. Get the 112-D face feature from the model (`val_a2m_forward(batch)['rec_face']`,
or `demo_v2_a2m_face.py`, which caches `_rec_face.npy` / `_audio.npy`), then render it here.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/render_face_flame.py \
      --rec_face experiments/v2_vis_a2m_face/_rec_face.npy \
      --audio    experiments/v2_vis_a2m_face/_audio.npy \
      --out      experiments/v2_vis_a2m_face/face.mp4 \
      --flame_dir model_files/FLAME2020
"""
import argparse
import numpy as np
from lom.render.flame_render import render_face_sequence


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rec_face', required=True, help='(.npy) face feature (N,112) = head6D + jaw6D + 100 expr')
    ap.add_argument('--audio', default=None, help='(.npy) 1-D waveform to mux (optional)')
    ap.add_argument('--out', required=True, help='output .mp4 path')
    ap.add_argument('--flame_dir', default='model_files/FLAME2020', help='FLAME2020 model dir (smplx.FLAME)')
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--sample_rate', type=int, default=16000)
    ap.add_argument('--image_size', type=int, default=512)
    ap.add_argument('--max_frames', type=int, default=None, help='cap number of rendered frames')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    rec_face = np.load(a.rec_face)
    audio = np.load(a.audio) if a.audio else None
    n = render_face_sequence(rec_face, a.flame_dir, a.out, audio=audio, fps=a.fps,
                             sample_rate=a.sample_rate, image_size=a.image_size,
                             max_frames=a.max_frames, device=a.device)
    print(f"face video -> {a.out}  ({n} frames)")


if __name__ == '__main__':
    main()
