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
    which = args.only.split(",") if args.only else ["a1", "a2", "a3", "a4", "a5", "a6"]
    if "a1" in which: a1(tr, va)
    if "a2" in which: a2(tr, va)
    if "a3" in which: a3()
    if "a4" in which: a4(tr)
    if "a5" in which: a5()
    if "a6" in which: a6(tr, va)
    print(f"\n  total {time.time()-t0:.0f}s\n")



# --------------------------------------------------------------------- A5
def a5():
    """The baseline the whole design has to beat."""
    import time as _t
    from tiny.model import FixedHeadClassifier
    hdr("A5  fixed-head baseline — is the flexible design worth it?")
    print("  same backbone, same loss, same seed. The only differences are that")
    print("  the options leave the input and the head becomes d -> k.\n")

    C = len(D.CATEGORIES)
    tr = build(20000, 0); va = build(5000, 99)
    slot = score(train(tr, "soft", "block"), va)

    # doc-only tensors for the plain classifier
    def docs(n, seed):
        ds = D.make_dataset(n, seed=seed)
        X = torch.tensor([list(d[:D.MAX_DOC]) + [D.PAD_ID]*(D.MAX_DOC-len(d[:D.MAX_DOC]))
                          for d, _, _ in ds])
        return X, torch.tensor([y for _, y, _ in ds]), \
               torch.tensor([p for _, _, p in ds], dtype=torch.float)

    Xtr, Ytr, Ptr = docs(20000, 0); Xva, Yva, Pva = docs(5000, 99)
    torch.manual_seed(1); random.seed(1)
    fh = FixedHeadClassifier(D.VOCAB_SIZE, C)
    opt = torch.optim.AdamW(fh.parameters(), lr=3e-3, weight_decay=0.01)
    n, bs = len(Xtr), 128
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=(n//bs)*10)
    for _ in range(10):
        perm = torch.randperm(n)
        for i in range(0, n-bs+1, bs):
            b = perm[i:i+bs]
            logp = fh(Xtr[b]).log_softmax(-1); p = logp.exp()
            ce = -(Ptr[b]*logp).sum(-1).mean()
            oh = F.one_hot(Ptr[b].argmax(-1), C).float()
            (ce + 0.3*((p-oh)**2).sum(-1).mean()).backward()
            torch.nn.utils.clip_grad_norm_(fh.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        pf = fh(Xva).softmax(-1); pred = pf.argmax(-1)
    fixed = dict(acc=(pred==Yva).float().mean().item(),
                 l1=(pf-Pva).abs().sum(-1).mean().item(),
                 ece=ece(pf.max(-1).values.numpy(), (pred==Yva).float().numpy()))

    # per-call latency, batch 1
    def lat(fn, *a, reps=300):
        with torch.no_grad():
            for _ in range(30): fn(*a)
            t0 = _t.time()
            for _ in range(reps): fn(*a)
        return (_t.time()-t0)/reps*1000
    sm = train(tr, "soft", "block")
    X1, ns1, S1, _, _, _ = va
    l_slot  = lat(lambda: sm(X1[:1], ns1, S1[:1]))
    l_fixed = lat(lambda: fh(Xva[:1]))

    n_slot  = sum(p.numel() for p in sm.parameters())
    n_fixed = sum(p.numel() for p in fh.parameters())
    print(f"  {'':26}{'slot head':>13}{'fixed head':>13}")
    for lab, a, b in [("accuracy", slot['acc'], fixed['acc']),
                      ("ECE", slot['ece'], fixed['ece']),
                      ("L1 to true posterior", slot['l1'], fixed['l1'])]:
        print(f"  {lab:26}{a:>13.4f}{b:>13.4f}")
    print(f"  {'Bayes ceiling':26}{D.bayes_ceiling(va[5]):>13.4f}{'':>13}")
    print(f"  {'sequence length':26}{D.MAX_DOC+len(D.CATEGORIES):>13}{D.MAX_DOC:>13}")
    print(f"  {'latency per call (ms)':26}{l_slot:>13.3f}{l_fixed:>13.3f}")
    print(f"  {'total parameters':26}{n_slot:>13,}{n_fixed:>13,}")
    print(f"  {'option sets per call':26}{'any':>13}{'fixed':>13}")
    print(f"\n  fixed head is {l_slot/l_fixed:.2f}x faster per call")


# --------------------------------------------------------------------- A6
def a6(tr, va):
    """Order sensitivity is a positional artefact. Remove the position
    difference between option slots and it should vanish, not shrink."""
    hdr("A6  fixing option-order sensitivity")
    print("  shuffling training data removed the model's INCENTIVE to key on")
    print("  slot position. It never removed its ABILITY to: each slot still")
    print("  carries a different position embedding. So give them all the same one.\n")
    C = len(D.CATEGORIES)
    X, ns, S, Y, P, raw = va

    def order_sensitivity(model, R=6):
        shifts, flips = [], []
        with torch.no_grad():
            base = model(X, ns, S).softmax(-1)
            for _ in range(R):
                q = torch.randperm(C)
                Xp = X.clone(); Xp[:, ns:] = Xp[:, ns:][:, q]
                pp = model(Xp, ns, S).softmax(-1)
                inv = torch.empty_like(q); inv[q] = torch.arange(C)
                pp = pp[:, inv]
                shifts.append((pp - base).abs().sum(-1).mean().item())
                flips.append((pp.argmax(-1) != base.argmax(-1)).float().mean().item())
        return float(np.mean(shifts)), float(np.mean(flips))

    def build_train(shared, shuf):
        torch.manual_seed(1); random.seed(1)
        Xt, nst, St, Yt, Pt, _ = tr
        m = TinySystemOne(D.VOCAB_SIZE, mask_mode="block", shared_opt_pos=shared)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
        n, bs = len(Xt), 128
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=(n//bs)*10)
        for _ in range(10):
            perm = torch.randperm(n)
            for i in range(0, n-bs+1, bs):
                b = perm[i:i+bs]
                xb, sb, tb = Xt[b], St[b], Pt[b]
                if shuf:
                    q = torch.randperm(C)
                    xb = xb.clone(); xb[:, nst:] = xb[:, nst:][:, q]; tb = tb[:, q]
                logp = m(xb, nst, sb).log_softmax(-1); p = logp.exp()
                ce = -(tb*logp).sum(-1).mean()
                oh = F.one_hot(tb.argmax(-1), C).float()
                (ce + 0.3*((p-oh)**2).sum(-1).mean()).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        return m

    print(f"  {'setup':<40}{'L1 shift':>11}{'flips':>9}{'acc':>9}{'ECE':>9}")
    for shared, shuf, lab in [(False, False, "per-slot positions, fixed order"),
                              (False, True,  "per-slot positions, shuffled"),
                              (True,  False, "SHARED position, fixed order"),
                              (True,  True,  "SHARED position, shuffled")]:
        m = build_train(shared, shuf)
        sh, fl = order_sensitivity(m)
        r = score(m, va)
        print(f"  {lab:<40}{sh:>11.4f}{fl:>8.1%}{r['acc']:>9.4f}{r['ece']:>9.4f}")
    print(f"\n  ceiling {D.bayes_ceiling(raw):.4f}. 0.0000 / 0.0% is exact order-invariance.")

if __name__ == "__main__":
    main()
