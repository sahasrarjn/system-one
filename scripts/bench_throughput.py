import time, torch
from transformers import AutoModelForCausalLM, AutoConfig

MODEL = "Qwen/Qwen3-0.6B"
dev = "mps"

print("loading", MODEL, "...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
cfg = AutoConfig.from_pretrained(MODEL)
model.to(dev); model.gradient_checkpointing_enable(); model.train()
print(f"loaded in {time.time()-t0:.1f}s")
n = sum(p.numel() for p in model.parameters())
print(f"params        {n/1e9:.3f}B   hidden {cfg.hidden_size}  layers {cfg.num_hidden_layers}  vocab {cfg.vocab_size}")
print(f"lm_head size  {cfg.hidden_size*cfg.vocab_size/1e6:.0f}M params  (tied={getattr(cfg,'tie_word_embeddings',None)})")

opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
B, T = 4, 512

def step():
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.mps.synchronize()

print("\nwarmup...", flush=True)
for _ in range(2): step()

N = 5
t0 = time.time()
for _ in range(N): step()
dt = (time.time() - t0) / N

tok_s = B * T / dt
flops = 6 * n * B * T / dt
print(f"\n=== MEASURED (batch {B} x seq {T}, grad checkpointing on) ===")
print(f"  step time     {dt*1000:.0f} ms")
print(f"  throughput    {tok_s:,.0f} tokens/sec")
print(f"  effective     {flops/1e12:.2f} TFLOP/s")
print(f"  peak memory   {torch.mps.current_allocated_memory()/1e9:.1f} GB")

for label, toks in [("week-one  20k ex x 400 tok x 3 ep", 20_000*400*3),
                    ("larger    50k ex x 400 tok x 3 ep", 50_000*400*3),
                    ("full      1.5M ex x 900 tok x 3 ep", 1_500_000*900*3)]:
    print(f"  {label:36s} -> {toks/tok_s/3600:6.1f} h")
