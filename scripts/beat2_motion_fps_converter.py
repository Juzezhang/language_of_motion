"""LoM v2 preprocessing: downsample BEAT2 SMPL-X motion from 30 fps -> 25 fps
(cubic interpolation), borrowed from ViBES. v2 uses 25 fps motion (vs v1's 30 fps)
so that, together with the 1x face tokenizer, the face token rate (25 tok/s) and the
4x body token rate (6.25 tok/s) match ViBES.

  python -m scripts.beat2_motion_fps_converter \
      --motion_folder datasets/BEAT2/beat_english_v2.0.0/smplxflame_30 \
      --output_dir    datasets/BEAT2/beat_english_v2.0.0/smplxflame_25
"""
import os
import argparse
import numpy as np
from os.path import join
from tqdm import tqdm
from scipy import interpolate


def main():
    parser = argparse.ArgumentParser(description="Downsample BEAT2 SMPL-X motion 30->25 fps.")
    parser.add_argument('--motion_folder', type=str, required=True,
                        help="[INPUT] BEAT2 SMPL-X folder at 30 fps (<BEAT2_ROOT>/smplxflame_30).")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="[OUTPUT] 25-fps version (<BEAT2_ROOT>/smplxflame_25).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    original_fps, target_fps = 30, 25
    done = 0
    for motion_file in tqdm(sorted(os.listdir(args.motion_folder)), desc="BEAT2 30->25 fps"):
        if not motion_file.endswith(".npz"):
            continue
        out_path = join(args.output_dir, motion_file)
        if os.path.exists(out_path):            # resume-friendly
            done += 1
            continue
        data = np.load(join(args.motion_folder, motion_file))
        poses, expressions, trans, betas = data['poses'], data['expressions'], data['trans'], data['betas']

        n0 = poses.shape[0]
        duration = (n0 - 1) / original_fps
        n1 = int(np.floor(duration * target_fps)) + 1
        t0 = np.linspace(0, duration, n0)
        t1 = np.linspace(0, duration, n1)

        def resample(seq):
            return interpolate.interp1d(t0, seq, axis=0, kind='cubic',
                                        bounds_error=False, fill_value="extrapolate")(t1)

        np.savez(out_path,
                 poses=resample(poses), expressions=resample(expressions), trans=resample(trans),
                 betas=betas, model=data['model'], gender=data['gender'],
                 mocap_frame_rate=np.array(25))
        done += 1
    print(f"Done: {done} files -> {args.output_dir}")


if __name__ == "__main__":
    main()
