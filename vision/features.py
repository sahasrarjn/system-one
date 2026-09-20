"""The frozen backbone, run once per example.

Why caching works at all: with the vision tower and the language model both
frozen, the hidden state at an option position is a fixed function of the
input. So we run the expensive part once, keep the vectors, and train the
probe on those. Training the head then costs seconds rather than another pass
over a 2B model.

The cost of that choice is that nothing below the probe can learn. Unfreezing
the merger is the next step up and needs a real GPU.
"""
import torch

from systemone.model.mask import batch_block_mask
from .packing import collate


def causal_mask_4d(lengths, n_total, dtype, device):
    """The stock mask, for the arm that does not get the block treatment."""
    B = len(lengths)
    neg = torch.finfo(dtype).min
    allow = torch.tril(torch.ones(n_total, n_total, dtype=torch.bool,
                                  device=device))[None].repeat(B, 1, 1)
    for i, L in enumerate(lengths):
        allow[i, :, L:] = False
    m = torch.where(allow, torch.zeros((), dtype=dtype, device=device),
                    torch.full((), neg, dtype=dtype, device=device))
    return m[:, None]


class FrozenEncoder:
    """Loads the VLM once and hands back hidden states at option positions."""

    def __init__(self, cfg, mask_mode: str = "block"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.cfg = cfg.resolved()
        self.mask_mode = mask_mode
        self.processor = AutoProcessor.from_pretrained(
            cfg.vlm, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.vlm, dtype=cfg.torch_dtype(), attn_implementation="sdpa",
        ).to(cfg.device).eval().requires_grad_(False)

        self.pad_id = self.processor.tokenizer.pad_token_id or 0
        self.hidden_size = self.model.config.text_config.hidden_size

    def _mask(self, n_states, lengths, T, dtype, device):
        if self.mask_mode == "block":
            return batch_block_mask(n_states, lengths, T, dtype, device)
        if self.mask_mode == "causal":
            return causal_mask_4d(lengths, T, dtype, device)
        raise ValueError(self.mask_mode)

    @torch.no_grad()
    def slot_states(self, packed_batch):
        """list of packed examples -> [B, K, d] float32 on CPU, plus slot_mask."""
        b = collate(packed_batch, self.pad_id)
        dev, dt = self.cfg.device, self.cfg.torch_dtype()

        input_ids = b["input_ids"].to(dev)
        pixel_values = b["pixel_values"].to(dev, dt)
        grid = b["image_grid_thw"].to(dev)
        attn_2d = b["attention_2d"].to(dev)
        mm_types = b["mm_types"].to(dev)
        T = input_ids.size(1)

        # Position ids come from the 2D mask, because the 3D rope helper reads
        # per-sample lengths off it. The 4D block mask goes to attention only.
        embeds = self.model.model.get_input_embeddings()(input_ids)
        position_ids = self.model.model.compute_3d_position_ids(
            input_ids=input_ids, inputs_embeds=embeds, image_grid_thw=grid,
            attention_mask=attn_2d, mm_token_type_ids=mm_types,
        )

        out = self.model.model(
            input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=grid,
            mm_token_type_ids=mm_types,
            attention_mask=self._mask(b["n_states"], b["lengths"], T, dt, dev),
            position_ids=position_ids, use_cache=False,
        )
        H = out.last_hidden_state                              # [B,T,d]

        idx = b["slot_idx"].to(dev).unsqueeze(-1).expand(-1, -1, H.size(-1))
        slots = torch.gather(H, 1, idx).float().cpu()          # [B,K,d]
        return slots, b["slot_mask"], H.float().cpu()
