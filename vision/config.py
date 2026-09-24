"""Model ids, device selection and the knobs that decide what a run costs."""
import os
from dataclasses import dataclass

import torch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class VisionConfig:
    # Qwen3-VL-2B is the smallest of the family. Its vision tower is
    # SigLIP2-Large (300M), which is the same encoder the bi-encoder baseline
    # uses, so the two arms differ in wiring rather than in what saw the pixels.
    vlm: str = "Qwen/Qwen3-VL-2B-Instruct"
    siglip: str = "google/siglip2-large-patch16-256"

    device: str = ""
    dtype: str = "bfloat16"

    # One visual token per 32x32 pixels (16px patches, merged 2x2). The cap is
    # in tokens, not pixels, because tokens are what the sequence length and
    # therefore the runtime are made of.
    max_visual_tokens: int = 256          # ~512x512
    min_visual_tokens: int = 64

    max_option_tokens: int = 16
    seed: int = 0

    cache_dir: str = "artifacts/vision"

    def torch_dtype(self):
        # bf16 needs Ampere or newer on CUDA, and MPS support for it is patchy.
        # Fall back to fp16 rather than silently running emulated bf16.
        if self.dtype == "bfloat16":
            if self.device == "mps":
                return torch.float16
            if self.device == "cuda" and not torch.cuda.is_bf16_supported():
                return torch.float16
        return getattr(torch, self.dtype)

    def resolved(self):
        if not self.device:
            self.device = os.environ.get("SO_DEVICE") or pick_device()
        return self

    @property
    def max_pixels(self) -> int:
        return self.max_visual_tokens * 32 * 32

    @property
    def min_pixels(self) -> int:
        return self.min_visual_tokens * 32 * 32
