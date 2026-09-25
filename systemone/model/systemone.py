"""Qwen3 backbone + block mask + slot head. No lm_head, no decoding."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .head import SlotHead
from .mask import batch_block_mask


class SystemOne(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # AutoModel, not AutoModelForCausalLM: this is how lm_head is dropped.
        # For Qwen3-0.6B that skips a 1024 x 151936 matmul at every position.
        # Compute only, not memory: tie_word_embeddings is true for this model,
        # so that matrix is embed_tokens reused transposed and stays resident
        # either way.
        self.backbone = AutoModel.from_pretrained(
            cfg.model_name, dtype=cfg.torch_dtype(),
            attn_implementation=cfg.attn_impl,
        )
        self.cfg = cfg
        self.head = SlotHead(self.backbone.config.hidden_size)
        if cfg.grad_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    @property
    def hidden_size(self):
        return self.backbone.config.hidden_size

    def encode(self, input_ids, n_states, lengths):
        mask = batch_block_mask(
            n_states, lengths, input_ids.size(1),
            dtype=self.backbone.dtype, device=input_ids.device,
        )
        out = self.backbone(input_ids=input_ids, attention_mask=mask,
                            use_cache=False)
        return out.last_hidden_state

    def forward(self, input_ids, slot_idx, slot_mask, n_states, lengths, **_):
        H = self.encode(input_ids, n_states, lengths)
        return self.head(H, slot_idx, slot_mask)

    @torch.no_grad()
    def decide(self, tokenizer, state, question, options):
        """Single-call inference. Returns the typed answer plus its distribution."""
        from .packing import pack_one, collate
        ids, ns, slots = pack_one(tokenizer, state, question, options,
                                  self.cfg.max_state_tokens,
                                  self.cfg.max_option_tokens)
        batch = collate([(ids, ns, slots, [0.0] * len(slots))],
                        tokenizer.pad_token_id or 0)
        dev = next(self.parameters()).device
        logits = self(**{k: (v.to(dev) if torch.is_tensor(v) else v)
                         for k, v in batch.items()})
        p = torch.softmax(logits[0, :len(options)].float(), -1)
        return {
            "value": options[int(p.argmax())],
            "p": p.tolist(),
            "confidence": float(confidence(p)),
        }


def confidence(p: torch.Tensor, n_options: torch.Tensor = None) -> torch.Tensor:
    """1 - normalised entropy. k-invariant, unlike max-probability:
    0.5 is decisive in a binary and near-uniform across ten options.

    `n_options` is the number of options ACTUALLY offered, per example. It is
    not optional in batched use: p is padded to the widest example in the
    batch, so p.shape[-1] is that width rather than this question's arity. A
    two-option question in a batch that also holds a thirteen-option one would
    be normalised by log(13), which destroys the k-invariance this function
    exists to provide.
    """
    H = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
    if n_options is None:
        n_options = torch.full(p.shape[:-1], p.shape[-1], device=p.device)
    n = n_options.to(p.dtype).clamp(min=2.0)
    return 1.0 - H / torch.log(n)


def load_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok
