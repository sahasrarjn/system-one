"""A System One model written from scratch, so every mechanism is visible.

Hand-rolled rather than HuggingFace on purpose: the whole point is to SEE
where the block mask enters attention, and that line is buried under three
abstraction layers in any real library.

~200K parameters. Trains in minutes on a CPU.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- the mask
def make_allow(mode: str, n_state: int, n_total: int, device=None) -> torch.Tensor:
    """Three mask variants, for the ablation.

      block  - ours. state self-contained; suffix bidirectional, sees state.
      causal - a normal LM mask. option j sees only options 1..j.
      full   - everything sees everything. Strictly more information than
               `block`, but the state now depends on the question, so nothing
               can be cached. If block ~= full, cacheability is free.
    """
    if mode == "block":
        return block_allow(n_state, n_total, device)
    if mode == "causal":
        return torch.tril(torch.ones(n_total, n_total, dtype=torch.bool, device=device))
    if mode == "full":
        return torch.ones(n_total, n_total, dtype=torch.bool, device=device)
    raise ValueError(mode)


def block_allow(n_state: int, n_total: int, device=None) -> torch.Tensor:
    """The one idea that makes 'encode once, answer N questions' work.

        state  -> state    ATTEND   bidirectional, self-contained
        state  -> suffix   MASKED   <-- so state KV never depends on the question
        suffix -> state    ATTEND
        suffix -> suffix   ATTEND   bidirectional, so options see each other

    A plain causal mask would break the last one (options couldn't compare).
    A fully bidirectional mask would break the second (state would depend on
    the question, and nothing could be cached).
    """
    a = torch.zeros(n_total, n_total, dtype=torch.bool, device=device)
    a[:n_state, :n_state] = True
    a[n_state:, :] = True
    return a


# ---------------------------------------------------------- the transformer
class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, allow):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)   # [B,h,T,T]
        # ---- THE LINE. everything else is a standard transformer. ----
        scores = scores.masked_fill(~allow[:, None], float("-inf"))
        attn = scores.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, allow):
        x = x + self.attn(self.ln1(x), allow)     # pre-norm
        return x + self.mlp(self.ln2(x))


class TinySystemOne(nn.Module):
    """Encoder + slot head. No lm_head, no decoding, no generation loop."""

    def __init__(self, vocab, d=64, depth=4, heads=4, max_len=64, mask_mode="block",
                 shared_opt_pos=False):
        super().__init__()
        self.mask_mode = mask_mode
        # Give every option slot the SAME position embedding. The slots then
        # differ only by their content, so permuting the options permutes the
        # scores and changes nothing else: order-invariance by construction,
        # rather than by hoping training teaches it.
        self.shared_opt_pos = shared_opt_pos
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(d)
        # The head: d -> 1, SHARED across slots. Parameter count is independent
        # of how many options the caller declares, which is the whole reason
        # the option set can change from one request to the next.
        self.probe = nn.Linear(d, 1)

    def encode(self, ids, n_state):
        B, T = ids.shape
        pos_ids = torch.arange(T, device=ids.device)
        if self.shared_opt_pos:
            pos_ids = pos_ids.clone()
            pos_ids[n_state:] = n_state          # every option slot, one position
        x = self.tok(ids) + self.pos(pos_ids)[None]
        allow = make_allow(self.mask_mode, n_state, T, ids.device)[None].expand(B, T, T)
        for blk in self.blocks:
            x = blk(x, allow)
        return self.ln(x)

    def forward(self, ids, n_state, slot_pos):
        """ids [B,T], slot_pos [B,K] -> logits [B,K]."""
        H = self.encode(ids, n_state)
        idx = slot_pos.unsqueeze(-1).expand(-1, -1, H.size(-1))
        return self.probe(torch.gather(H, 1, idx)).squeeze(-1)


def confidence(p: torch.Tensor) -> torch.Tensor:
    """1 - normalised entropy. k-invariant: 0.5 is decisive in a binary but
    near-uniform across ten options, so plain max-probability is not
    comparable across questions of different arity."""
    k = p.shape[-1]
    H = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
    return 1.0 - H / math.log(k)


class FixedHeadClassifier(nn.Module):
    """The ordinary alternative, for comparison.

    Same embeddings, same blocks, same depth. Two differences:
      - the answer options are NOT in the input, so the sequence is shorter
      - one d -> k head on a pooled document vector, so k is fixed at build time

    This is what you would write if your label set never changed. It is the
    baseline the System One design has to justify itself against.
    """

    def __init__(self, vocab, n_classes, d=64, depth=4, heads=4, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)      # d -> k, fixed

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))[None]
        allow = torch.ones(T, T, dtype=torch.bool, device=ids.device)[None].expand(B, T, T)
        for blk in self.blocks:
            x = blk(x, allow)
        return self.head(self.ln(x).mean(1))     # mean-pool the document
