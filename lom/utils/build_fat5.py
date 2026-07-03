"""Build a FAT5 (catie-aq/flashT5) language model initialized from a HuggingFace
flan-t5 checkpoint, with the fixes needed to use it as a drop-in seq2seq LM:

  * convert HF T5 weight keys -> FAT5 layout (0 missing / 0 unexpected for flan-t5);
  * seed the (untied) lm_head from the shared embedding;
  * bind get/set_output_embeddings so resize_token_embeddings also resizes lm_head;
  * bind set_input_embeddings so resize propagates to the encoder/decoder stacks
    (otherwise generate() indexes a stale embedding and triggers a CUDA assert);
  * set valid pad/eos/decoder-start token ids.

FAT5 is vendored under third_party/flashT5 (no pip package).
"""
import os, re, sys, types
import torch
from transformers import T5ForConditionalGeneration

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_FAT5_PATH = os.path.join(_REPO_ROOT, "third_party", "flashT5")


def _ensure_fat5_on_path():
    if _FAT5_PATH not in sys.path:
        sys.path.insert(0, _FAT5_PATH)


def _convert_key(k):
    for pat, rep in [
        (r"\.layer\.\d+\.SelfAttention\.q", ".self_attention_layer.self_attention.Wq"),
        (r"\.layer\.\d+\.SelfAttention\.k", ".self_attention_layer.self_attention.Wk"),
        (r"\.layer\.\d+\.SelfAttention\.v", ".self_attention_layer.self_attention.Wv"),
        (r"\.layer\.\d+\.SelfAttention\.o", ".self_attention_layer.self_attention.o"),
        (r"\.layer\.\d+\.EncDecAttention\.q", ".cross_attention_layer.cross_attention.Wq"),
        (r"\.layer\.\d+\.EncDecAttention\.k", ".cross_attention_layer.cross_attention.Wk"),
        (r"\.layer\.\d+\.EncDecAttention\.v", ".cross_attention_layer.cross_attention.Wv"),
        (r"\.layer\.\d+\.EncDecAttention\.o", ".cross_attention_layer.cross_attention.o"),
        (r"\.layer\.\d+\.SelfAttention\.relative_attention_bias\.",
         ".self_attention_layer.self_attention.pe_encoding.relative_attention_bias."),
    ]:
        k = re.sub(pat, rep, k)
    k = k.replace(".layer.0.layer_norm.", ".self_attention_layer.layer_norm.")
    k = k.replace(".layer.1.layer_norm.",
                  ".ff_layer.layer_norm." if "encoder" in k else ".cross_attention_layer.layer_norm.")
    k = k.replace(".layer.2.layer_norm.", ".ff_layer.layer_norm.")
    k = re.sub(r"\.layer\.\d+\.DenseReluDense\.", ".ff_layer.", k)
    k = k.replace(".wi_", ".act.wi_")
    return k


def build_fat5_from_hf(model_path, attention_type="triton"):
    """Return a FlashT5ForConditionalGeneration warm-started from the HF T5 at
    `model_path`. attention_type: 'triton' (flash, default) | 'fa2' | 'ref'."""
    _ensure_fat5_on_path()
    from src.model.configuration_flash_t5 import FlashT5Config
    from src.model.modeling_flash_t5 import FlashT5ForConditionalGeneration

    hf = T5ForConditionalGeneration.from_pretrained(model_path)
    hc = hf.config
    cfg = FlashT5Config(
        vocab_size=hc.vocab_size, d_model=hc.d_model, d_kv=hc.d_kv, d_ff=hc.d_ff,
        num_layers=hc.num_layers, num_decoder_layers=hc.num_decoder_layers or hc.num_layers,
        num_heads=hc.num_heads,
        relative_attention_num_buckets=hc.relative_attention_num_buckets,
        relative_attention_max_distance=hc.relative_attention_max_distance,
        attention_type=attention_type, position_encoding_type="t5",
        max_sequence_length=2048,
        use_glu_mlp=getattr(hc, "is_gated_act", True),
        use_gelu_act="gelu" in getattr(hc, "dense_act_fn", "gelu"),
        pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
        z_loss=0.0, label_smoothing=0.0,   # FAT5 loss multiplies by these; None -> TypeError
    )
    cfg.tie_word_embeddings = False
    model = FlashT5ForConditionalGeneration(cfg)

    sd = {_convert_key(k): v for k, v in hf.state_dict().items()}
    if "lm_head.weight" not in sd and "shared.weight" in sd:
        sd["lm_head.weight"] = sd["shared.weight"].clone()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, \
        f"FAT5 weight conversion mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"

    # --- patches so it behaves as a resizable HF seq2seq LM ---
    model.get_output_embeddings = types.MethodType(lambda self: self.lm_head, model)
    model.set_output_embeddings = types.MethodType(
        lambda self, new: setattr(self, "lm_head", new), model)

    def _set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = value
        self.decoder.embed_tokens = value
    model.set_input_embeddings = types.MethodType(_set_input_embeddings, model)

    return model
