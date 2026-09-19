"""Four ablations. Each tests a claim that was asserted but never measured.

    python -m tiny.ablate

A1  mask variant      - does the block mask actually earn its place?
A2  option order      - is scoring order-invariant, and does shuffling in
                        training matter?
A3  ambiguity sweep   - does calibration survive across task difficulty, or
                        did one operating point flatter us?
A4  unseen subsets    - do option sets never seen in training still work?
"""
import argparse, itertools, math, random, time
import numpy as np
import torch
import torch.nn.functional as F

from tiny import data as D
from tiny.model import TinySystemOne, confidence


# ----------------------------------------------------------------- helpers
def build(n, seed, signal=0.20, opts=None):
    ds = D.make_dataset(n, signal=signal, seed=seed)
    opts = opts or list(range(len(D.CATEGORIES)))
    packed = [D.pack(d, opts) for d, _, _ in ds]
    return (torch.tensor([p[0] for p in packed]), packed[0][1],
            torch.tensor([p[2] for p in packed]),
            torch.tensor([y for _, y, _ in ds]),
            torch.tensor([p for _, _, p in ds], dtype=torch.float), ds)


def train(tr, kind="soft", mask="block", epochs=10, seed=1, shuffle_opts=False):
    torch.manual_seed(seed); random.seed(seed)
    X, ns, S, Y, P, _ = tr
    C = len(D.CATEGORIES)
    model = TinySystemOne(D.VOCAB_SIZE, mask_mode=mask)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    target = P if kind == "soft" else F.one_hot(Y, C).float()
    n, bs = len(X), 128
    steps = (n // bs) * epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)

    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            b = perm[i:i + bs]
            xb, sb, tb = X[b], S[b], target[b]
            if shuffle_opts:
                # permute the option tokens (and their slots/targets) per batch
                q = torch.randperm(C)
                xb = xb.clone(); xb[:, ns:] = xb[:, ns:][:, q]
                tb = tb[:, q]
            logits = model(xb, ns, sb)
            logp = logits.log_softmax(-1); p = logp.exp()
            ce = -(tb * logp).sum(-1).mean()
            oh = F.one_hot(tb.argmax(-1), C).float()
            loss = ce + 0.3 * ((p - oh) ** 2).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    return model


@torch.no_grad()
def score(model, va):
    X, ns, S, Y, P, _ = va
    p = model(X, ns, S).softmax(-1)
    pred = p.argmax(-1)
    return dict(p=p, acc=(pred == Y).float().mean().item(),
                l1=(p - P).abs().sum(-1).mean().item(),
                ece=ece(p.max(-1).values.numpy(), (pred == Y).float().numpy()))


def ece(p_top, correct, bins=12):
    o = np.argsort(p_top); out = 0.0; n = len(p_top)
    for b in np.array_split(o, bins):
        if len(b): out += len(b) / n * abs(p_top[b].mean() - correct[b].mean())
    return float(out)


def hdr(t):
    print("\n" + "=" * 72); print("  " + t); print("=" * 72)


# --------------------------------------------------------------------- A1
def a1(tr, va):
    hdr("A1  mask variant — does the block mask earn its place?")
    print("  claim: block ≈ full (so cacheability is free) and both > causal\n")
    print(f"  {'mask':<10}{'accuracy':>11}{'ECE':>10}{'L1 to truth':>14}   cacheable?")
    rows = {}
    for m in ("causal", "block", "full"):
        t0 = time.time()
        r = score(train(tr, "soft", m), va)
        rows[m] = r
        cache = {"causal": "yes", "block": "yes", "full": "NO"}[m]
        print(f"  {m:<10}{r['acc']:>11.4f}{r['ece']:>10.4f}{r['l1']:>14.4f}   {cache}"
              f"   ({time.time()-t0:.0f}s)")
    print(f"\n  ceiling   {D.bayes_ceiling(va[5]):>11.4f}")
    d_bc = rows['block']['acc'] - rows['causal']['acc']
    d_fb = rows['full']['acc'] - rows['block']['acc']
    print(f"\n  block − causal = {d_bc:+.4f} accuracy")
    print(f"  full  − block  = {d_fb:+.4f} accuracy   <- the price of cacheability")
    return rows


# --------------------------------------------------------------------- A2
def a2(tr, va):
    hdr("A2  option order — is scoring order-invariant?")
    print("  the tiny model trains on a FIXED option order, and position")
    print("  embeddings differ per slot, so invariance is NOT automatic.\n")
    C = len(D.CATEGORIES)
    X, ns, S, Y, P, _ = va
    print(f"  {'trained with':<20}{'mask':<10}{'mean L1 shift':>16}{'argmax flips':>15}")
    for shuf in (False, True):
        for mask in ("block", "causal"):
            model = train(tr, "soft", mask, shuffle_opts=shuf)
            shifts, flips, R = [], [], 4
            with torch.no_grad():
                base = model(X, ns, S).softmax(-1)
                for _ in range(R):
                    q = torch.randperm(C)
                    Xp = X.clone(); Xp[:, ns:] = Xp[:, ns:][:, q]
                    pp = model(Xp, ns, S).softmax(-1)
                    inv = torch.empty_like(q); inv[q] = torch.arange(C)
                    pp = pp[:, inv]                 # un-permute
                    shifts.append((pp - base).abs().sum(-1).mean().item())
                    flips.append((pp.argmax(-1) != base.argmax(-1)).float().mean().item())
            lbl = "shuffled options" if shuf else "fixed order"
            print(f"  {lbl:<20}{mask:<10}{np.mean(shifts):>16.4f}{np.mean(flips):>14.1%}")
    print("\n  0.0000 / 0.0% would mean the answer does not depend on where an")
    print("  option sits in the sequence.")


# --------------------------------------------------------------------- A3
def a3():
    hdr("A3  ambiguity sweep — does calibration survive task difficulty?")
    print("  one operating point could be luck. Five, across a 2x range of")
    print("  ceiling, is a characterisation.\n")
    print(f"  {'signal':>7}{'ceiling':>10}{'acc(hard)':>11}{'acc(soft)':>11}"
          f"{'ECE(hard)':>11}{'ECE(soft)':>11}{'L1(hard)':>10}{'L1(soft)':>10}")
    for sig in (0.10, 0.15, 0.20, 0.25, 0.30):
        tr = build(12000, 0, signal=sig); va = build(4000, 99, signal=sig)
        ceil = D.bayes_ceiling(va[5])
        rh = score(train(tr, "hard", "block", epochs=8), va)
        rs = score(train(tr, "soft", "block", epochs=8), va)
        print(f"  {sig:>7.2f}{ceil:>10.3f}{rh['acc']:>11.4f}{rs['acc']:>11.4f}"
              f"{rh['ece']:>11.4f}{rs['ece']:>11.4f}{rh['l1']:>10.4f}{rs['l1']:>10.4f}")


# --------------------------------------------------------------------- A4
def a4(tr):
    hdr("A4  unseen option subsets — are option sets really free?")
    print("  trained ONLY on the full 4-way question. Evaluated on subsets it")
    print("  has never been asked. Exact target = full posterior renormalised.\n")
    model = train(tr, "soft", "block")
    C = len(D.CATEGORIES)
    raw = D.make_dataset(3000, seed=99)
    print(f"  {'k':>3}  {'option subset':<40}{'accuracy':>10}{'L1 to truth':>13}")
    agg = {}
    for k in (2, 3, 4):
        accs, l1s = [], []
        for subset in itertools.combinations(range(C), k):
            ids, ns, slots, tgt = [], None, [], []
            for d, y, p in raw:
                if y not in subset:      # restricted question only defined here
                    continue
                i, n_s, sl = D.pack(d, list(subset))
                ids.append(i); ns = n_s; slots.append(sl)
                sub = np.array([p[c] for c in subset]); sub = sub / sub.sum()
                tgt.append((sub, list(subset).index(y)))
            if not ids:
                continue
            with torch.no_grad():
                pr = model(torch.tensor(ids), ns, torch.tensor(slots)).softmax(-1)
            T = torch.tensor(np.array([t[0] for t in tgt]), dtype=torch.float)
            gold = torch.tensor([t[1] for t in tgt])
            a = (pr.argmax(-1) == gold).float().mean().item()
            l = (pr - T).abs().sum(-1).mean().item()
            accs.append(a); l1s.append(l)
            if k == 2:
                names = "+".join(D.CATEGORIES[c][:4] for c in subset)
                print(f"  {k:>3}  {names:<40}{a:>10.4f}{l:>13.4f}")
        agg[k] = (float(np.mean(accs)), float(np.mean(l1s)))
    print()
    for k, (a, l) in agg.items():
        tag = "  <- trained on this one" if k == 4 else ""
        print(f"  k={k} mean over all subsets   acc {a:.4f}   L1 {l:.4f}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    t0 = time.time()
    tr = build(20000, 0); va = build(5000, 99)
    which = args.only.split(",") if args.only else ["a1", "a2", "a3", "a4"]
    if "a1" in which: a1(tr, va)
    if "a2" in which: a2(tr, va)
    if "a3" in which: a3()
    if "a4" in which: a4(tr)
    print(f"\n  total {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
