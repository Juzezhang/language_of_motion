"""LoM v2 face tokenization (borrowed from ViBES): tokenize BEAT2 SMPL-X face into
1x face tokens with ViBES's face VQ (VQVAEConvZeroDSUS1_PaperVersion + face.ckpt).

Unlike the v1 compositional face (106D = jaw6d+expr, 4x-downsampled via lom_vq), v2's
face is ViBES's 112D = 6D head (poses[:,45:48]) + 6D jaw (poses[:,66:69]) + 100 expr,
tokenized at 1x (one token per frame -> 25 tok/s at 25 fps), codebook 512. Verified to
match ViBES's own precomputed BEAT2 face tokens at ~97%.

  python -m scripts.get_face_code_v2 \
      --motion_folder datasets/BEAT2/beat_english_v2.0.0/smplxflame_25 \
      --output_dir    datasets/BEAT2/beat_english_v2.0.0/TOKENS_DS4_v2/face \
      --face_ckpt     model_files/pretrained_cpt/face/face.ckpt
"""
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from lom.archs.lom_vq import VQVAEConvZeroDSUS1_PaperVersion
from lom.utils.rotation_conversions import axis_angle_to_6d_np

FACE_DIM = 112  # 6D head + 6D jaw + 100 expression (ViBES)


def build_face_112(npz):
    d = np.load(npz)
    poses = d["poses"]
    n = poses.shape[0]
    head6d = axis_angle_to_6d_np(poses[:, 45:48]).reshape(n, 6)   # head joint (SMPL-X joint 15)
    jaw6d = axis_angle_to_6d_np(poses[:, 66:69]).reshape(n, 6)    # jaw joint (SMPL-X joint 22)
    exp = d["expressions"][:, :100]
    return np.concatenate([head6d, jaw6d, exp], axis=1).astype(np.float32)


def load_face_vq(ckpt, device):
    m = VQVAEConvZeroDSUS1_PaperVersion(vae_layer=3, code_num=512, vae_test_dim=FACE_DIM, codebook_size=512)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k[len("vae_face."):]: v for k, v in sd.items() if k.startswith("vae_face.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"face.ckpt mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
    return m.eval().to(device)


def main():
    parser = argparse.ArgumentParser(description="GLM-style 1x face tokenization (LoM v2, ViBES face VQ).")
    parser.add_argument("--motion_folder", type=str, required=True, help="BEAT2 smplxflame_25 folder.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir for face token .npy.")
    parser.add_argument("--face_ckpt", type=str, default="model_files/pretrained_cpt/face/face.ckpt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_face_vq(args.face_ckpt, args.device)

    files = sorted(f for f in os.listdir(args.motion_folder) if f.endswith(".npz"))
    done = 0
    for fn in tqdm(files, desc="v2 face tokenize (1x)"):
        out_path = os.path.join(args.output_dir, fn[:-4] + ".npy")
        if os.path.exists(out_path):
            done += 1
            continue
        face = build_face_112(os.path.join(args.motion_folder, fn))
        x = torch.from_numpy(face)[None].to(args.device)
        with torch.no_grad():
            codes = model.map2index(x).reshape(1, -1).to("cpu").numpy()   # (1, T) at 1x
        np.save(out_path, codes)
        done += 1
    print(f"Done: {done}/{len(files)} face token files -> {args.output_dir}")


if __name__ == "__main__":
    main()
