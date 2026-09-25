import argparse, json, math, os, random, time
import numpy as np, torch
from torch.utils.data import DataLoader

from systemone.config import Config
from systemone.data.dataset import (DecisionDataset, make_collate,
                                    TokenBudgetSampler)
from systemone.model.systemone import SystemOne, load_tokenizer, confidence
from systemone.model.losses import decision_loss


def to_device(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, dev, amp=None, max_batches=0):
    model.eval()
    P, C, K, NLL, n = [], [], [], 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        b = to_device(batch, dev)
        with torch.autocast(dev, dtype=amp) if amp else _null():
            logits = model(**b)
        sm = b["slot_mask"]
        logits = logits.float().masked_fill(~sm, -1e9)
        p = torch.softmax(logits, -1)
        gold = b["target"].argmax(-1)
        lp = torch.log_softmax(logits, -1)
        NLL += -lp.gather(-1, gold[:, None]).squeeze(-1).sum().item()
        n += len(gold)
        P.append(p.max(-1).values.cpu().numpy())
        C.append((p.argmax(-1) == gold).float().cpu().numpy())
        K.append(confidence(p, sm.sum(-1)).cpu().numpy())
    model.train()
    return (np.concatenate(P), np.concatenate(C), np.concatenate(K),
            NLL / max(1, n))


import contextlib


@contextlib.contextmanager
def _null():
    yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="artifacts/data")
    ap.add_argument("--out", default="artifacts/runs/run1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-state-tokens", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-batch", type=int, default=16,
                    help="hard cap on examples per batch, whatever the budget")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="token budget per batch; 0 uses a fixed batch size")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="drop grads and Adam state for the (tied) embedding")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="trade ~30%% speed for a large drop in activation memory")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=120,
                    help="batches per mid-run eval; 0 = whole val set")
    ap.add_argument("--limit-steps", type=int, default=0, help="smoke test")
    ap.add_argument("--no-save", action="store_true", help="skip writing weights")
    args = ap.parse_args()

    cfg = Config(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                 max_state_tokens=args.max_state_tokens,
                 grad_accum=args.grad_accum, load_dtype="float32",
                 grad_checkpointing=args.grad_checkpointing)
    dev = cfg.resolve_device()
    amp = cfg.torch_amp_dtype() if cfg.amp_ok(dev) else None
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    print(f"device={dev} weights=float32 autocast={amp} model={cfg.model_name}")
    if dev == "cuda" and amp is None:
        print("  WARNING: bf16 autocast unavailable on this GPU (pre-Ampere); "
              "running fp32, which will be slow")

    tok = load_tokenizer(cfg)
    model = SystemOne(cfg).to(dev)
    n_body = sum(p.numel() for p in model.backbone.parameters())
    print(f"backbone {n_body/1e6:.0f}M params | head "
          f"{sum(p.numel() for p in model.head.parameters())} params | no lm_head")

    coll = make_collate(tok.pad_token_id or 0)
    tr = DecisionDataset(f"{args.data_dir}/train.jsonl", tok, cfg)
    va = DecisionDataset(f"{args.data_dir}/val.jsonl", tok, cfg,
                         shuffle_options=False)
    if args.max_tokens:
        # Sequence length varies ~20x across the arity ladder, so a fixed batch
        # size is sized for the average and OOMs on the tail.
        strn, svan = (TokenBudgetSampler(d.approx_lengths(), args.max_tokens,
                                         max_batch=args.max_batch, seed=cfg.seed)
                      for d in (tr, va))
        dl = DataLoader(tr, batch_sampler=strn, collate_fn=coll)
        dv = DataLoader(va, batch_sampler=svan, collate_fn=coll)
        nb = len(strn)
        print(f"train {len(tr):,} questions | val {len(va):,} | "
              f"{nb:,} batches at <={args.max_tokens} tokens "
              f"(mean {len(tr)/max(1,nb):.1f} examples)")
    else:
        dl = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, collate_fn=coll)
        dv = DataLoader(va, batch_size=cfg.batch_size, collate_fn=coll)
        print(f"train {len(tr):,} questions | val {len(va):,}")

    if args.freeze_embeddings:
        # embed_tokens is 151,936 x 1,024 = 26% of this model's parameters, and
        # it is tied, so it is also the output projection. Freezing it drops its
        # gradient and both Adam moments: ~1.9GB that a classification
        # fine-tune has little use for.
        model.backbone.get_input_embeddings().requires_grad_(False)
    trainable = [q for q in model.parameters() if q.requires_grad]
    # fused=True does the update in place with almost no scratch space.
    # The default foreach=True path fuses across the parameter LIST instead,
    # allocating temporaries proportional to total parameter size: measured at
    # ~7GiB of transient peak on 440M fp32 params, on a batch of 1,488 tokens
    # whose activations are a few hundred MB. That fixed spike, invisible to
    # batch-size tuning, is what made three separate OOM diagnoses wrong.
    opt = torch.optim.AdamW(trainable, lr=cfg.lr,
                            weight_decay=cfg.weight_decay,
                            fused=(dev == "cuda"))
    n_train = sum(q.numel() for q in trainable)
    print(f"trainable {n_train/1e6:.0f}M of "
          f"{sum(q.numel() for q in model.parameters())/1e6:.0f}M params")
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"  after model+optimiser setup: "
              f"{torch.cuda.memory_allocated()/2**30:.2f} GiB resident")
    total = (len(dl) * cfg.epochs // cfg.grad_accum) if not args.limit_steps \
        else args.limit_steps
    warm = max(1, int(total * cfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        s / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))))

    step, t0, best = 0, time.time(), float("inf")
    micro = 0
    hist = []
    for ep in range(cfg.epochs):
        for batch in dl:
            b = to_device(batch, dev)
            with torch.autocast(dev, dtype=amp) if amp else _null():
                logits = model(**b)
            loss, parts = decision_loss(logits, b["target"], b["slot_mask"],
                                        cfg.brier_weight)
            (loss / cfg.grad_accum).backward()
            micro += 1
            if micro % cfg.grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1

            if step % 20 == 0:
                mem = ""
                if dev == "cuda":
                    mem = (f" | mem {torch.cuda.memory_allocated()/2**30:.1f}"
                           f"/{torch.cuda.max_memory_allocated()/2**30:.1f} GiB"
                           f" B={b['input_ids'].shape[0]}xT={b['input_ids'].shape[1]}")
                print(f"ep{ep} step {step}/{total} loss {loss.item():.4f} "
                      f"ce {parts['ce']:.4f} brier {parts['brier']:.4f} "
                      f"({(time.time()-t0)/step:.2f}s/step){mem}", flush=True)

            if args.eval_every and step % args.eval_every == 0:
                _, c, _, nll = evaluate(model, dv, dev, amp, args.eval_batches)
                hist.append({"step": step, "val_nll": nll, "val_acc": float(c.mean())})
                flag = ""
                if nll < best:
                    best = nll
                    flag = "  <- best"
                    if not args.no_save:
                        torch.save({"head": model.head.state_dict(),
                                    "step": step, "val_nll": nll},
                                   f"{args.out}/best_head.pt")
                print(f"  [eval] step {step}  val nll {nll:.4f}  "
                      f"acc {c.mean():.3f}{flag}", flush=True)
                json.dump(hist, open(f"{args.out}/history.json", "w"), indent=2)

            if args.limit_steps and step >= args.limit_steps:
                break
        if args.limit_steps and step >= args.limit_steps:
            break

    # a smoke run should not pay for a full-val pass at the end
    p, c, k, nll = evaluate(model, dv, dev, amp,
                            args.eval_batches if args.limit_steps else 0)
    # val loader is unshuffled, so dataset order lines up with prediction order
    np.savez(f"{args.out}/val_preds.npz", p_top=p, correct=c, conf=k,
             source=np.array(va.sources[:len(p)]),
             arity=np.array([len(va.items[i][1].options) for i in range(len(p))]))
    hist.append({"step": step, "val_nll": nll, "val_acc": float(c.mean()),
                 "final": True})
    json.dump(hist, open(f"{args.out}/history.json", "w"), indent=2)
    if not args.no_save:
        torch.save({"head": model.head.state_dict(), "cfg": cfg.__dict__},
                   f"{args.out}/head.pt")
        model.backbone.save_pretrained(f"{args.out}/backbone")
        tok.save_pretrained(f"{args.out}/backbone")
    print(f"val acc {c.mean():.3f}  nll {nll:.4f}  |  "
          f"preds -> {args.out}/val_preds.npz")


if __name__ == "__main__":
    main()
