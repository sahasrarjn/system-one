from dataclasses import dataclass, field


@dataclass
class Config:
    # --- model ---
    model_name: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    device: str = "auto"              # auto -> cuda | mps | cpu
    attn_impl: str = "sdpa"           # "eager" is slower but most predictable

    # --- packing ---
    max_state_tokens: int = 1024      # week-one. Jev's spec is 32k.
    max_option_tokens: int = 32
    max_options: int = 255            # the documented Jev ceiling

    # --- training ---
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    brier_weight: float = 0.3         # weight on the proper scoring rule
    grad_checkpointing: bool = False  # peak mem is low; off is ~30% faster
    seed: int = 0

    # --- data ---
    data_dir: str = "artifacts/data"
    out_dir: str = "artifacts/runs"
    cfpb_per_product: int = 3000      # stratify! CFPB is 80% credit reporting
    civil_n: int = 40000

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def torch_dtype(self):
        import torch
        return getattr(torch, self.dtype)
