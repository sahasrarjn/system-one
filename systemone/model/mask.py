"""The block attention mask.

Layout of every packed sequence:

    [ state .................... ][ question | options+slots ]
      0                    n_state                         T

Four quadrants, three of which attend:

    state  -> state    ATTEND   bidirectional, self-contained
    state  -> suffix   MASKED   <-- the whole point
    suffix -> state    ATTEND
    suffix -> suffix   ATTEND   bidirectional, so options see each other

The masked quadrant is what makes the state's keys and values independent of
the question. That is what lets one encode serve N questions, and it is
exactly the property a globally-bidirectional mask would destroy.
"""
import torch


def block_allow(n_state: int, n_total: int, device=None) -> torch.Tensor:
    """Boolean [T, T]; True = this query may attend to this key."""
    allow = torch.zeros(n_total, n_total, dtype=torch.bool, device=device)
    allow[:n_state, :n_state] = True     # state block, bidirectional
    allow[n_state:, :] = True            # suffix sees state and itself
    return allow


def block_mask_4d(n_state, n_total, dtype, device=None, pad_len: int = None):
    """Additive float mask shaped [1, 1, T, T] for HF attention modules.

    0.0 where attention is allowed, dtype-min where it is not.
    `pad_len` masks trailing padding positions as keys.
    """
    allow = block_allow(n_state, n_total, device)
    if pad_len is not None and pad_len < n_total:
        allow[:, pad_len:] = False
    neg = torch.finfo(dtype).min
    m = torch.where(allow, torch.zeros((), dtype=dtype, device=device),
                    torch.full((), neg, dtype=dtype, device=device))
    return m[None, None]


def batch_block_mask(n_states, lengths, n_total, dtype, device=None):
    """[B, 1, T, T] for a batch whose examples have different state lengths."""
    return torch.cat([
        block_mask_4d(ns, n_total, dtype, device, pad_len=L)
        for ns, L in zip(n_states, lengths)
    ], dim=0)
