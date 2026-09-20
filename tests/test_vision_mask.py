"""Does the block mask survive contact with a real Qwen3-VL forward pass?

Three properties, on a randomly-initialised tiny model so this costs nothing:

  1. the 4D mask is accepted at all (transformers can silently ignore masks it
     does not recognise, which would make every result meaningless);
  2. the state block's hidden states do not change when the question and
     options change  <- the property the whole cache argument rests on;
  3. a globally-bidirectional mask DOES change them, so property 2 is a
     consequence of the mask and not of something else.
"""
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_vl import (
    Qwen3VLConfig, Qwen3VLForConditionalGeneration,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig, Qwen3VLVisionConfig,
)

from vision.packing import pack_one
from vision.features import FrozenEncoder
from vision.config import VisionConfig

MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def tiny_model():
    """Same architecture, ~1000x smaller. Vocab must stay full: the real
    tokenizer emits ids up to 151k and an embedding lookup would go out of
    bounds otherwise."""
    text = Qwen3VLTextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        vocab_size=151936, rope_scaling={"mrope_interleaved": True,
                                         "mrope_section": [8, 4, 4],
                                         "rope_type": "default"},
    )
    vision = Qwen3VLVisionConfig(
        hidden_size=64, intermediate_size=128, depth=4, num_heads=2,
        patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
        out_hidden_size=64, deepstack_visual_indexes=[1, 2, 3],
        num_position_embeddings=2304,
    )
    cfg = Qwen3VLConfig(text_config=text, vision_config=vision)
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(cfg).eval().requires_grad_(False)


def build(mask_mode):
    enc = FrozenEncoder.__new__(FrozenEncoder)
    enc.cfg = VisionConfig(device="cpu", dtype="float32").resolved()
    enc.cfg.device = "cpu"
    enc.mask_mode = mask_mode
    enc.processor = AutoProcessor.from_pretrained(
        MODEL, min_pixels=64 * 32 * 32, max_pixels=128 * 32 * 32)
    enc.model = tiny_model()
    enc.pad_id = enc.processor.tokenizer.pad_token_id or 0
    enc.hidden_size = 64
    return enc


def main():
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))

    enc = build("block")
    a = pack_one(enc.processor, img, "what animal is this?", ["cat", "dog"])
    b = pack_one(enc.processor, img, "is the photograph taken outdoors, and if "
                 "so in what season?", ["yes", "no", "cannot tell", "partly"])
    n_state = a["n_state"]
    assert b["n_state"] == n_state, "same image must give the same state length"
    assert len(a["input_ids"]) != len(b["input_ids"]), "suffixes must differ"
    print(f"state = {n_state} tokens (image + markers)")
    print(f"suffix A = {len(a['input_ids']) - n_state} tokens, "
          f"suffix B = {len(b['input_ids']) - n_state} tokens")

    _, _, Ha = enc.slot_states([a])
    _, _, Hb = enc.slot_states([b])
    print("1. the 4D block mask was accepted            OK")

    drift = (Ha[0, :n_state] - Hb[0, :n_state]).abs().max().item()
    print(f"2. state drift when the question changes     {drift:.3e}")
    assert drift < 1e-5, f"state depends on the suffix: {drift}"

    enc.mask_mode = "full_bidirectional"
    enc._mask = lambda ns, L, T, dt, dev: torch.zeros(
        len(L), 1, T, T, dtype=dt, device=dev)
    _, _, Fa = enc.slot_states([a])
    _, _, Fb = enc.slot_states([b])
    fdrift = (Fa[0, :n_state] - Fb[0, :n_state]).abs().max().item()
    print(f"3. same test, globally bidirectional mask    {fdrift:.3e}")
    assert fdrift > 1e-4, "control failed: the mask is being ignored entirely"

    print("\nall three hold. the state is cacheable because of the mask,")
    print("and the control shows the mask is what is doing it.")


if __name__ == "__main__":
    main()
