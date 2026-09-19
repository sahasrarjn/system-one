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
        # For Qwen3-0.6B that is a 1024 x 151936 = 156M-param projection we
        # never build and never run.
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


def confidence(p: torch.Tensor) -> torch.Tensor:
    """1 - normalised entropy. k-invariant, unlike max-probability:
    0.5 is decisive in a binary and near-uniform across ten options."""
    k = p.shape[-1]
    if k < 2:
        return torch.ones_like(p[..., 0])
    H = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
    import math
    return 1.0 - H / math.log(k)


def load_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok
