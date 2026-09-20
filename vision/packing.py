"""Turn (image, question, options) into one packed sequence.

The layout is identical to the text pipeline:

    [ <vision_start> image tokens <vision_end> (context) ][ question | options ]
      0 ................................... n_state ................... T

Everything before n_state is the state block. The only difference from
`systemone/model/packing.py` is that part of the state is an image, and the
processor is what decides how many tokens that costs.
"""
from typing import List

import torch


def image_state_ids(processor, image, context: str = ""):
    """Token ids for the state block, plus the pixel tensors that go with it.

    The processor expands the single <|image_pad|> placeholder into however
    many visual tokens this image's resolution implies, so we never have to
    compute the count ourselves.
    """
    text = "<|vision_start|><|image_pad|><|vision_end|>"
    if context:
        text += context
    enc = processor(text=[text], images=[image], return_tensors="pt")
    # mm_token_type_ids marks which positions are visual. M-RoPE needs it to
    # lay out the (t, h, w) grid, so it has to survive packing.
    return (enc["input_ids"][0].tolist(), enc["pixel_values"],
            enc["image_grid_thw"], enc["mm_token_type_ids"][0].tolist())


def pack_one(processor, image, question: str, options: List[str],
             context: str = "", max_option_tokens: int = 16):
    """Returns a dict with input_ids, n_state, slots and the pixel tensors.

    The slot for option j is the position of that option's LAST token, which
    is the same convention the text pipeline uses: every option ends at a
    readout position and no vocabulary surgery is needed.
    """
    tok = processor.tokenizer
    state_ids, pixel_values, grid, mm_types = image_state_ids(
        processor, image, context)
    n_state = len(state_ids)

    ids = list(state_ids)
    ids += tok(f"\nQ: {question}", add_special_tokens=False)["input_ids"]

    slots = []
    for opt in options:
        opt_ids = tok(f"\n- {opt}", add_special_tokens=False)["input_ids"]
        opt_ids = opt_ids[:max_option_tokens]
        if not opt_ids:
            opt_ids = tok("\n- ?", add_special_tokens=False)["input_ids"]
        ids.extend(opt_ids)
        slots.append(len(ids) - 1)

    assert n_state > 0, "empty state block"
    assert all(s >= n_state for s in slots), "an option slot landed in the state"
    mm_types = list(mm_types) + [0] * (len(ids) - n_state)   # suffix is all text
    return dict(input_ids=ids, n_state=n_state, slots=slots, mm_types=mm_types,
                pixel_values=pixel_values, image_grid_thw=grid)


def collate(packed: List[dict], pad_id: int):
    """Pad a list of packed examples into batched tensors.

    Images have different token counts, so both the sequence length and the
    state boundary vary within a batch. The mask is built per example for
    exactly that reason.
    """
    T = max(len(p["input_ids"]) for p in packed)
    K = max(len(p["slots"]) for p in packed)
    B = len(packed)

    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    mm_types = torch.zeros((B, T), dtype=torch.long)
    slot_idx = torch.zeros((B, K), dtype=torch.long)
    slot_mask = torch.zeros((B, K), dtype=torch.bool)
    attn_2d = torch.zeros((B, T), dtype=torch.long)
    n_states, lengths = [], []

    for i, p in enumerate(packed):
        ids = p["input_ids"]
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        mm_types[i, :len(ids)] = torch.tensor(p["mm_types"], dtype=torch.long)
        attn_2d[i, :len(ids)] = 1
        slot_idx[i, :len(p["slots"])] = torch.tensor(p["slots"], dtype=torch.long)
        slot_mask[i, :len(p["slots"])] = True
        n_states.append(p["n_state"])
        lengths.append(len(ids))

    return dict(
        input_ids=input_ids, slot_idx=slot_idx, slot_mask=slot_mask,
        attention_2d=attn_2d, mm_types=mm_types,
        n_states=n_states, lengths=lengths,
        pixel_values=torch.cat([p["pixel_values"] for p in packed], dim=0),
        image_grid_thw=torch.cat([p["image_grid_thw"] for p in packed], dim=0),
    )
