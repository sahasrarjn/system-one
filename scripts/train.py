import argparse, json, math, os, random, time
import numpy as np, torch
from torch.utils.data import DataLoader

from systemone.config import Config
from systemone.data.dataset import DecisionDataset, make_collate
from systemone.model.systemone import SystemOne, load_tokenizer, confidence
from systemone.model.losses import decision_loss


def to_device(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    P, C, K = [], [], []
    for batch in loader:
        b = to_device(batch, dev)
        logits = model(**b)
        p = torch.softmax(logits.float(), -1)
        pred, gold = p.argmax(-1), b["target"].argmax(-1)
        P.append(p.max(-1).values.cpu().numpy())
        C.append((pred == gold).float().cpu().numpy())
        K.append(confidence(p).cpu().numpy())
    model.train()
    return np.concatenate(P), np.concatenate(C), np.concatenate(K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="artifacts/data")
    ap.add_argument("--out", default="artifacts/runs/run1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-state-tokens", type=int, default=512)
    ap.add_argument("--limit-steps", type=int, default=0, help="smoke test")
    ap.add_argument("--no-save", action="store_true", help="skip writing weights")
    args = ap.parse_args()

    cfg = Config(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                 max_state_tokens=args.max_state_tokens)
    dev = cfg.resolve_device()
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    print(f"device={dev} dtype={cfg.dtype} model={cfg.model_name}")

    tok = load_tokenizer(cfg)
    model = SystemOne(cfg).to(dev)
    n_body = sum(p.numel() for p in model.backbone.parameters())
    print(f"backbone {n_body/1e6:.0f}M params | head "
          f"{sum(p.numel() for p in model.head.parameters())} params | no lm_head")

    coll = make_collate(tok.pad_token_id or 0)
    tr = DecisionDataset(f"{args.data_dir}/train.jsonl", tok, cfg)
    va = DecisionDataset(f"{args.data_dir}/val.jsonl", tok, cfg,
                         shuffle_options=False)
    dl = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, collate_fn=coll)
    dv = DataLoader(va, batch_size=cfg.batch_size, collate_fn=coll)
    print(f"train {len(tr):,} questions | val {len(va):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    total = len(dl) * cfg.epochs if not args.limit_steps else args.limit_steps
    warm = max(1, int(total * cfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        s / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))))

    step, t0 = 0, time.time()
    for ep in range(cfg.epochs):
        for batch in dl:
            loss, parts = decision_loss(model(**to_device(batch, dev)),
                                        to_device(batch, dev)["target"],
                                        batch["slot_mask"].to(dev),
                                        cfg.brier_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                print(f"ep{ep} step {step}/{total} loss {loss.item():.4f} "
                      f"ce {parts['ce']:.4f} brier {parts['brier']:.4f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
            if args.limit_steps and step >= args.limit_steps:
                break
        if args.limit_steps and step >= args.limit_steps:
            break

    p, c, k = evaluate(model, dv, dev)
    np.savez(f"{args.out}/val_preds.npz", p_top=p, correct=c, conf=k)
    if not args.no_save:
        torch.save({"head": model.head.state_dict(), "cfg": cfg.__dict__},
                   f"{args.out}/head.pt")
        model.backbone.save_pretrained(f"{args.out}/backbone")
        tok.save_pretrained(f"{args.out}/backbone")
    print(f"val acc {c.mean():.3f}  |  preds -> {args.out}/val_preds.npz")


if __name__ == "__main__":
    main()
