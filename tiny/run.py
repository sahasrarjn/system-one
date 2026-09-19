"""Train a System One model from scratch and demonstrate every Jev property.

    python -m tiny.run

Runs in a couple of minutes on a CPU. The ablation in the middle is the
lesson: the SAME architecture and the SAME loss, trained on hard labels
versus soft targets, produce very differently calibrated models.
"""
import argparse, math, random, time
import numpy as np
import torch
import torch.nn.functional as F

from tiny import data as D
from tiny.model import TinySystemOne, confidence


def build(n, seed):
    ds = D.make_dataset(n, seed=seed)
    ids, ns, slots = zip(*[D.pack(d, list(range(len(D.CATEGORIES)))) for d, _, _ in ds])
    return (torch.tensor(ids), ns[0], torch.tensor(slots),
            torch.tensor([y for _, y, _ in ds]),
            torch.tensor([p for _, _, p in ds], dtype=torch.float))


def train(target_kind, tr, va, epochs, seed, brier_w=0.3, quiet=False):
    torch.manual_seed(seed)
    X, ns, S, Y, P = tr
    model = TinySystemOne(D.VOCAB_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    target = P if target_kind == "soft" else F.one_hot(Y, len(D.CATEGORIES)).float()
    n, bs = len(X), 128
    steps = (n // bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)

    step = 0
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            b = perm[i:i + bs]
            logits = model(X[b], ns, S[b])
            logp = logits.log_softmax(-1)
            p = logp.exp()
            ce = -(target[b] * logp).sum(-1).mean()
            onehot = F.one_hot(target[b].argmax(-1), len(D.CATEGORIES)).float()
            brier = ((p - onehot) ** 2).sum(-1).mean()
            loss = ce + brier_w * brier
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
        if not quiet and (ep + 1) % 2 == 0:
            print(f"    epoch {ep+1:>2}/{epochs}  loss {loss.item():.4f}")
    return model


@torch.no_grad()
def assess(model, va):
    X, ns, S, Y, P = va
    p = model(X, ns, S).softmax(-1)
    pred = p.argmax(-1)
    return {
        "p": p.numpy(),
        "correct": (pred == Y).float().numpy(),
        "p_top": p.max(-1).values.numpy(),
        "conf": confidence(p).numpy(),
        "acc": (pred == Y).float().mean().item(),
        # only possible with synthetic data: distance to the TRUE posterior
        "l1_to_truth": (p - P).abs().sum(-1).mean().item(),
        "mean_stated": p.max(-1).values.mean().item(),
    }


def ece(p_top, correct, bins=12):
    order = np.argsort(p_top)
    out, n = 0.0, len(p_top)
    for b in np.array_split(order, bins):
        if len(b):
            out += len(b) / n * abs(p_top[b].mean() - correct[b].mean())
    return out


def diagram(p_top, correct, bins=10, width=42):
    order = np.argsort(p_top)
    print("      stated                                            observed")
    for b in np.array_split(order, bins):
        if not len(b):
            continue
        c, a = p_top[b].mean(), correct[b].mean()
        row = [" "] * width
        row[min(width - 1, int(c * (width - 1)))] = "."
        row[min(width - 1, int(a * (width - 1)))] = "#"
        print(f"      {c:.2f}  {''.join(row)}  {a:.2f}")
    print("            (. = what the model claimed, # = what actually happened)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--val-n", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    random.seed(0); np.random.seed(0)

    print("=" * 66)
    print("  A System One model, from scratch")
    print("=" * 66)
    tr = build(args.train_n, seed=0)
    va = build(args.val_n, seed=99)
    raw_va = D.make_dataset(args.val_n, seed=99)
    ceiling = D.bayes_ceiling(raw_va)

    m = TinySystemOne(D.VOCAB_SIZE)
    print(f"\n  parameters   {sum(p.numel() for p in m.parameters()):,}")
    print(f"  head         {sum(p.numel() for p in m.probe.parameters()):,} "
          f"(d->1, shared across slots -> option count is free)")
    print(f"  train / val  {args.train_n:,} / {args.val_n:,}")
    print(f"  Bayes ceiling {ceiling:.3f}  <- no model can beat this; the rest")
    print(f"                             is irreducible ambiguity, not error")

    print("\n" + "-" * 66)
    print("  ABLATION: same architecture, same loss, different targets")
    print("-" * 66)
    results = {}
    for kind, label in [("hard", "hard labels (the sampled category)"),
                        ("soft", "soft targets (the true posterior)")]:
        print(f"\n  training on {label}")
        t0 = time.time()
        model = train(kind, tr, va, args.epochs, seed=1)
        r = assess(model, va)
        r["secs"] = time.time() - t0
        r["ece"] = ece(r["p_top"], r["correct"])
        results[kind] = (model, r)
        print(f"    done in {r['secs']:.0f}s")

    print(f"\n  {'':22s}{'hard':>12}{'soft':>12}   {'':4}")
    rows = [("accuracy", "acc", "higher"), ("mean stated prob", "mean_stated", ""),
            ("ECE (empirical)", "ece", "lower"),
            ("L1 to TRUE posterior", "l1_to_truth", "lower")]
    for name, key, _ in rows:
        h, s = results["hard"][1][key], results["soft"][1][key]
        better = "soft" if (key in ("ece", "l1_to_truth")) == (s < h) else "hard"
        print(f"  {name:22s}{h:>12.4f}{s:>12.4f}   <- {better}")
    print(f"  {'Bayes ceiling':22s}{ceiling:>12.3f}{ceiling:>12.3f}")

    for kind in ("hard", "soft"):
        print(f"\n  reliability — {kind} targets")
        r = results[kind][1]
        diagram(r["p_top"], r["correct"])

    model = results["soft"][0]
    print("\n" + "-" * 66)
    print("  THE JEV PROPERTIES")
    print("-" * 66)

    # 1. arbitrary option sets, decided at call time
    doc, y, truth = raw_va[0]
    print(f"\n  1. option set is declared per call, not baked into the head")
    print(f"     doc: {D.decode(doc)[:56]}...")
    for subset in ([0, 1, 2, 3], [0, 3], [1, 2]):
        ids, ns, slots = D.pack(doc, subset)
        with torch.no_grad():
            p = model(torch.tensor([ids]), ns, torch.tensor([slots])).softmax(-1)[0]
        names = [D.CATEGORIES[c] for c in subset]
        pick = names[int(p.argmax())]
        print(f"     k={len(subset)}  {str(names):<44} -> {pick:<10} "
              f"conf {float(confidence(p)):.2f}")

    # 2. type safety is structural
    print(f"\n  2. type safety is structural, not learned")
    ok = all(
        D.CATEGORIES[s[int(model(torch.tensor([D.pack(d, s)[0]]), D.MAX_DOC,
             torch.tensor([D.pack(d, s)[2]])).argmax())]] in [D.CATEGORIES[i] for i in s]
        for d, _, _ in raw_va[:200] for s in ([0, 1, 2], [1, 3])
    )
    print(f"     200 docs x 2 option sets, every answer inside its declared set: {ok}")
    print(f"     it cannot be otherwise: the softmax has no mass to give elsewhere")

    # 3. encode once, answer many
    print(f"\n  3. encode once, answer N questions")
    ids1, ns, s1 = D.pack(doc, [0, 1])
    with torch.no_grad():
        t0 = time.time()
        for _ in range(200):
            model(torch.tensor([ids1]), ns, torch.tensor([s1]))
        t1 = (time.time() - t0) / 200
        ids4, ns, s4 = D.pack(doc, [0, 1, 2, 3])
        t0 = time.time()
        for _ in range(200):
            model(torch.tensor([ids4]), ns, torch.tensor([s4]))
        t4 = (time.time() - t0) / 200
    print(f"     2 options {t1*1000:.2f} ms   4 options {t4*1000:.2f} ms"
          f"   ratio {t4/t1:.2f}x for 2x the questions")
    print(f"     the state is encoded once; extra options are nearly free")
    print()


if __name__ == "__main__":
    main()
