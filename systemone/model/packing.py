"""Turn (state, question, options) into token ids plus slot positions.

Tokenises piecewise rather than formatting one big string, so slot positions
are exact instead of recovered from character offsets.
"""
from typing import List, Tuple


def pack_one(tokenizer, state: str, question: str, options: List[str],
             max_state_tokens: int = 1024, max_option_tokens: int = 32
             ) -> Tuple[List[int], int, List[int]]:
    """Returns (input_ids, n_state, slot_positions).

    The slot for option j is the position of that option's LAST token: every
    option ends at a readout position, and no vocabulary surgery is needed.
    """
    state_ids = tokenizer(state, add_special_tokens=False)["input_ids"]
    state_ids = state_ids[:max_state_tokens]
    n_state = len(state_ids)

    ids = list(state_ids)
    ids += tokenizer(f"\nQ: {question}", add_special_tokens=False)["input_ids"]

    slots = []
    for opt in options:
        opt_ids = tokenizer(f"\n- {opt}", add_special_tokens=False)["input_ids"]
        opt_ids = opt_ids[:max_option_tokens]
        if not opt_ids:                      # pathological empty option
            opt_ids = tokenizer("\n- ?", add_special_tokens=False)["input_ids"]
        ids.extend(opt_ids)
        slots.append(len(ids) - 1)

    assert n_state > 0, "empty state after tokenisation"
    assert all(s >= n_state for s in slots), "slot landed inside the state block"
    return ids, n_state, slots


def collate(batch, pad_id: int):
    """batch: list of (ids, n_state, slots, target). Pads to the longest."""
    import torch
    T = max(len(b[0]) for b in batch)
    K = max(len(b[2]) for b in batch)
    B = len(batch)

    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    slot_idx = torch.zeros((B, K), dtype=torch.long)
    slot_mask = torch.zeros((B, K), dtype=torch.bool)
    target = torch.zeros((B, K), dtype=torch.float)
    n_states, lengths = [], []

    for i, (ids, ns, slots, tgt) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        slot_idx[i, :len(slots)] = torch.tensor(slots, dtype=torch.long)
        slot_mask[i, :len(slots)] = True
        target[i, :len(tgt)] = torch.tensor(tgt, dtype=torch.float)
        n_states.append(ns)
        lengths.append(len(ids))

    return dict(input_ids=input_ids, slot_idx=slot_idx, slot_mask=slot_mask,
                target=target, n_states=n_states, lengths=lengths)
