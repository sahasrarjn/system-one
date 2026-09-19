"""The slot-readout head.

One SHARED linear probe, d -> 1, applied at every slot position. Parameters
are independent of the option count, which is what lets the option set change
from request to request. For Qwen3-0.6B (d=1024) the head is 1,025 params.
"""
import torch
import torch.nn as nn


class SlotHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.probe = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.probe.weight, std=0.02)
        nn.init.zeros_(self.probe.bias)

    def forward(self, H: torch.Tensor, slot_idx: torch.Tensor,
                slot_mask: torch.Tensor) -> torch.Tensor:
        """H [B,T,d], slot_idx [B,K], slot_mask [B,K] -> logits [B,K]."""
        idx = slot_idx.unsqueeze(-1).expand(-1, -1, H.size(-1))
        gathered = torch.gather(H, 1, idx)                    # [B,K,d]
        logits = self.probe(gathered.float()).squeeze(-1)     # [B,K]
        return logits.masked_fill(~slot_mask, torch.finfo(logits.dtype).min)
