"""GLM-4-Voice audio tokenizer — the v2 audio frontend (replaces HuBERT).

Uses GLM-4-Voice's Whisper-VQ speech tokenizer (vendored in third_party/glm4voice).
  * codebook size : 16384  (token ids in [0, 16383])
  * frame rate    : 12.5 tokens / second
  * input         : 16 kHz mono audio

NOTE: requires transformers>=4.43 (EncoderDecoderCache). Run this in the GLM-4-Voice /
v2 environment, NOT the turbot5-pinned v1 env (transformers==4.42). The LoM model itself
only consumes the precomputed integer token ids, so tokenization is an offline step.
"""
import os
import sys
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GLM_PATH = os.path.join(_REPO_ROOT, "third_party", "glm4voice")
DEFAULT_TOKENIZER_PATH = os.path.join(_REPO_ROOT, "model_files", "glm-4-voice-tokenizer")

AUDIO_CODEBOOK_SIZE = 16384   # vs HuBERT's 500 in v1
TOKEN_FPS = 12.5


def load_glm4voice_tokenizer(model_path=DEFAULT_TOKENIZER_PATH, device="cuda"):
    """Load the GLM-4-Voice Whisper-VQ encoder + feature extractor."""
    if _GLM_PATH not in sys.path:
        sys.path.insert(0, _GLM_PATH)
    from transformers import WhisperFeatureExtractor
    from speech_tokenizer.modeling_whisper import WhisperVQEncoder
    model = WhisperVQEncoder.from_pretrained(model_path).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    return model, feature_extractor


@torch.no_grad()
def tokenize_audio(model, feature_extractor, audio, sample_rate, device="cuda"):
    """audio: 1D float32 array/tensor (mono) or (C, T). Returns list[int] of token ids in [0, 16383]."""
    import torchaudio
    a = torch.as_tensor(audio, dtype=torch.float32)
    if a.ndim > 1:
        a = a[0]                              # first channel -> mono
    if sample_rate != 16000:
        a = torchaudio.functional.resample(a, sample_rate, 16000)
    a = a.cpu().numpy()

    pooling = model.config.pooling_kernel_size or 1
    stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling * feature_extractor.hop_length

    # 30s chunks (avoids GPU OOM on long clips), same as GLM-4-Voice's reference impl.
    chunks, t = [], 0
    while t * 16000 < len(a):
        chunks.append(a[t * 16000:(t + 30) * 16000])
        t += 30
    if not chunks:
        chunks = [a]

    tokens = []
    for s in range(0, len(chunks), 128):
        feats = feature_extractor(chunks[s:s + 128], sampling_rate=16000,
                                  return_attention_mask=True, return_tensors="pt",
                                  padding="longest", pad_to_multiple_of=stride).to(device)
        st = model(**feats).quantized_token_ids
        am = feats.attention_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]][:, ::pooling]
        for i in range(len(st)):
            tokens.extend(st[i][am[i].bool()].tolist())
    return tokens


def tokenize_wav(model, feature_extractor, wav_path, device="cuda"):
    """Tokenize a .wav file -> list[int]."""
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)   # (T, C)
    return tokenize_audio(model, feature_extractor, audio[:, 0], sr, device)
