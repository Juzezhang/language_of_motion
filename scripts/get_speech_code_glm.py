"""LoM v2 audio tokenization: tokenize a folder of .wav files into GLM-4-Voice audio
tokens (one .npy of int64 token ids per wav). This replaces the HuBERT-based
get_speech_code_* scripts for v2 (codebook 16384, 12.5 tokens/s, vs HuBERT's 500).

Run in the GLM-4-Voice / v2 environment (transformers>=4.43), e.g.:

  conda activate GLM-4-Voice
  python -m scripts.get_speech_code_glm \
      --wav_folder datasets/BEAT2/beat_english_v2.0.0/wave16k \
      --output_dir datasets/BEAT2/beat_english_v2.0.0/audios_token_glm

Then point the dataset config's `code_path_audio` at `audios_token_glm`.
"""
import os
import argparse
import numpy as np
from os.path import join
from tqdm import tqdm

from lom.utils.glm4voice_tokenizer import (
    load_glm4voice_tokenizer, tokenize_wav, DEFAULT_TOKENIZER_PATH,
)


def main():
    parser = argparse.ArgumentParser(description="GLM-4-Voice audio tokenization (LoM v2).")
    parser.add_argument("--wav_folder", type=str, required=True,
                        help="Folder of .wav files to tokenize.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output folder for the .npy token arrays.")
    parser.add_argument("--speech_tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH,
                        help="GLM-4-Voice tokenizer (local dir or HF id THUDM/glm-4-voice-tokenizer).")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, feature_extractor = load_glm4voice_tokenizer(args.speech_tokenizer, args.device)

    # walk recursively; handle .wav (BEAT2) and .flac (LibriSpeech); mirror input structure.
    exts = (".wav", ".flac")
    files = []
    for root, _, fnames in os.walk(args.wav_folder):
        for fn in fnames:
            if fn.lower().endswith(exts):
                files.append(os.path.join(root, fn))
    files.sort()

    done = 0
    for fp in tqdm(files, desc="GLM-4-Voice tokenizing"):
        rel = os.path.relpath(fp, args.wav_folder)
        out_path = join(args.output_dir, os.path.splitext(rel)[0] + ".npy")
        if os.path.exists(out_path):          # resume-friendly
            done += 1
            continue
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tokens = tokenize_wav(model, feature_extractor, fp, args.device)
        np.save(out_path, np.array(tokens, dtype=np.int64))
        done += 1
    print(f"Done: {done}/{len(files)} files -> {args.output_dir}")


if __name__ == "__main__":
    main()
