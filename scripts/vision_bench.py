"""Does encode-once-ask-many actually pay, in wall clock, on a real image?

The token arithmetic says forty questions about one image should cost roughly
two encodes instead of forty. That is an argument, not a measurement, and the
two differ whenever fixed overhead dominates -- which is exactly what happened
when I predicted a 20% speedup for a fixed-head text baseline and measured 3%.

So: same model, same image, same forty questions, two execution strategies.

  naive   full forward per question, which is what a causal model must do
  cached  encode the state once, then run each suffix against the saved KV

The block mask is what makes the second one legal: the state never attends to
the suffix, so its keys and values do not depend on which question follows.
"""
import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.config import VisionConfig
from vision.features import FrozenEncoder
from vision.packing import pack_one
from systemone.model.mask import block_mask_4d

QUESTIONS = [
    ("What kind of animal is this?", ["a cat", "a dog", "neither"]),
    ("Is the animal indoors?", ["yes", "no", "cannot tell"]),
    ("What colour is its coat?", ["black", "white", "ginger", "grey", "mixed"]),
    ("Is more than one animal visible?", ["yes", "no"]),
    ("Is the animal looking at the camera?", ["yes", "no", "cannot tell"]),
]


def neg(dt):
    return torch.finfo(dt).min


@torch.no_grad()
def run(n_questions=40, image_px=512):
    from PIL import Image
    import numpy as np

    cfg = VisionConfig().resolved()
    enc = FrozenEncoder(cfg, mask_mode="block")
    dev, dt = cfg.device, cfg.torch_dtype()
    m = enc.model.model

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (image_px, image_px, 3),
                                       dtype=np.uint8))
    qs = [QUESTIONS[i % len(QUESTIONS)] for i in range(n_questions)]
    packs = [pack_one(enc.processor, img, q, o) for q, o in qs]
    n_state = packs[0]["n_state"]
    suffix_lens = [len(p["input_ids"]) - n_state for p in packs]
    print(f"device {dev}  state {n_state} visual+marker tokens  "
          f"{n_questions} questions  suffix {min(suffix_lens)}-{max(suffix_lens)} tokens")

    def sync():
        if dev == "mps":
            torch.mps.synchronize()
        elif dev == "cuda":
            torch.cuda.synchronize()

    # ---- warm up, so the first kernel launch is not charged to arm one
    enc.slot_states([packs[0]]); sync()

    # ---- arm 1: a full forward for every question
    t0 = time.perf_counter()
    for p in packs:
        enc.slot_states([p])
    sync()
    naive = time.perf_counter() - t0

    # ---- arm 2: encode the state once, then replay suffixes against its KV
    from transformers import DynamicCache
    t0 = time.perf_counter()
    ids = torch.tensor([packs[0]["input_ids"][:n_state]], device=dev)
    mmt = torch.tensor([packs[0]["mm_types"][:n_state]], device=dev)
    px = packs[0]["pixel_values"].to(dev, dt)
    grid = packs[0]["image_grid_thw"].to(dev)
    embeds = m.get_input_embeddings()(ids)
    pos = m.compute_3d_position_ids(input_ids=ids, inputs_embeds=embeds,
                                    image_grid_thw=grid,
                                    attention_mask=torch.ones_like(ids),
                                    mm_token_type_ids=mmt)
    cache = DynamicCache(config=enc.model.config.text_config)
    state_mask = torch.zeros(1, 1, n_state, n_state, dtype=dt, device=dev)
    m(input_ids=ids, pixel_values=px, image_grid_thw=grid, mm_token_type_ids=mmt,
      attention_mask=state_mask, position_ids=pos, past_key_values=cache,
      use_cache=True)
    sync()
    encode_once = time.perf_counter() - t0

    # The state's KV: computed once, reused for every question. This is the
    # object the block mask makes question-independent.
    state_kv = [(l.keys.clone(), l.values.clone()) for l in cache.layers]

    t0 = time.perf_counter()
    for p in packs:
        sfx = p["input_ids"][n_state:]
        S = len(sfx)
        c = DynamicCache(config=enc.model.config.text_config)
        for i, (k, v) in enumerate(state_kv):
            c.update(k.clone(), v.clone(), i)
        sid = torch.tensor([sfx], device=dev)
        semb = m.get_input_embeddings()(sid)
        spos = (pos[..., -1:] + torch.arange(1, S + 1, device=dev)
                .view(1, 1, -1).expand(pos.shape[0], 1, -1))
        # suffix sees all of the state and all of itself
        smask = torch.zeros(1, 1, S, n_state + S, dtype=dt, device=dev)
        m.language_model(inputs_embeds=semb, attention_mask=smask,
                         position_ids=spos, past_key_values=c, use_cache=True)
    sync()
    replay = time.perf_counter() - t0

    print(f"\n  naive   {naive:6.2f}s   {naive/n_questions*1000:6.0f} ms/question")
    print(f"  cached  {encode_once + replay:6.2f}s   "
          f"{(encode_once + replay)/n_questions*1000:6.0f} ms/question"
          f"   (encode {encode_once:.2f}s + replay {replay:.2f}s)")
    print(f"\n  speedup {naive / (encode_once + replay):.1f}x over "
          f"{n_questions} questions")


if __name__ == "__main__":
    run(n_questions=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
