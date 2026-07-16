# a2f_decoder — Decoder-only 4-part Audio-to-Motion (inference)

Inference harness for the **Language of Motion** decoder-only 4-part a2m checkpoints
(Qwen3-1.7B / Qwen3-0.6B). Weights are on HuggingFace
([`JuzeZhang/LoM-a2m-4part`](https://huggingface.co/JuzeZhang/LoM-a2m-4part)); this code turns audio
tokens into a full-body SMPL-X talking + gesturing avatar.

The model generates one causal stream and each block is decoded by its VQ:

```
[BOS] audio-tokens [SEP] [FACE] face [UPPER] upper [LOWER] lower [HAND] hand [EOS]
```

## Dependencies

- `transformers >= 4.49` (Qwen3 support; 5.x also works), `torch >= 2.2`, `soundfile`, `numpy`
- **`model_files/`** (download separately — VQ decoders + body models, not redistributed):
  - `model_files/pretrained_cpt/face/face_lom.ckpt` — face VQ (512, 112-D)
  - `model_files/pretrained_cpt/lom_vq_ds/lom_vq.ckpt` — body VQ (upper/lower/hand, 256 each)
  - `model_files/pretrained_cpt/VAE_Global_from_Lower54/vibes_global.ckpt` — global-translation VAE
  - `model_files/smplx_models/` — SMPL-X body model (smpl-x.is.tue.mpg.de)
  - FLAME2020 (for the optional face close-up render)
- **GLM-4-Voice tokenizer** to turn a `.wav` into audio tokens (12.5 tok/s, 16384-way). See the ViBES
  speech pipeline ([`JuzeZhang/YouTube_Talking`](https://huggingface.co/datasets/JuzeZhang/YouTube_Talking)).
  You can also pass pre-tokenized `.npy` audio tokens directly.

## Run

```bash
# external audio: GLM audio-token .npy -> 4 motion streams -> SMPL-X (+ optional wav for A/V)
python a2f_decoder/infer_4part.py \
    --ckpt qwen3-1.7B_4part.pt \
    --audio_tokens my_audio_glm.npy --audio_wav my_audio.wav \
    --out out/ --save_only
```

Output per clip: SMPL-X `.npy` (pose + expr + trans), a face `.npy`, and the trimmed `.wav`.
Rendering (SMPL-X body via Blender + FLAME face close-up) is a separate CPU step.

> `--beat2_names` / `--tfhp` reproduce the paper's held-out demos from the *internal* datasets; external
> users should use **`--audio_tokens <file.npy>`**, which needs none of the training data.

## Checkpoint format

Each `.pt` = `torch.load` → `{model, cfg, vocab, step, val_loss, val_acc}`. `cfg.pretrained` names the
Qwen3 backbone; `vocab` carries the token offsets. `load_decoder` rebuilds the model (keeps the Qwen3 text
vocab, resizes embeddings, loads the state dict).
