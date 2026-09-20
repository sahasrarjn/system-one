"""End-to-end: cache frozen features, train both heads, compare.

    python scripts/vision_run.py cache   --split trainval --limit 1500
    python scripts/vision_run.py cache   --split test     --limit 800
    python scripts/vision_run.py train
    python scripts/vision_run.py report

Split into steps because the cache step is the only expensive one, and it only
has to happen once per (model, mask, dataset) combination.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.config import VisionConfig
from vision.data import build_examples, load_pets, NONE_OPTION
from vision.packing import pack_one
from vision.baseline import SiglipFeatures, BiEncoderHead, DiagHead, is_none_matrix
from systemone.model.head import SlotHead
from systemone.model.losses import decision_loss
from systemone.evaluation.metrics import ece, brier_decomposition

OUT = "artifacts/vision"


def cache(args):
    cfg = VisionConfig().resolved()
    os.makedirs(OUT, exist_ok=True)
    ds, classes = load_pets(split=args.split)
    labels = [ds[i][1] for i in range(len(ds))]
    ex = build_examples(labels, classes, seed=cfg.seed)
    if args.limit:
        ex = ex[:args.limit]
    K = max(e.k for e in ex)
    print(f"{args.split}: {len(ex)} questions, max {K} options, "
          f"{len(classes)} breeds, device={cfg.device}")

    # ---- arm B: bi-encoder (cheap, do it first so a crash costs little)
    sig = SiglipFeatures(cfg)
    uniq = sorted({o for e in ex for o in e.options})
    lab_emb = sig.label_embeds([l for l in uniq])
    lab_ix = {l: i for i, l in enumerate(uniq)}
    img_emb = []
    t0 = time.time()
    for i in range(0, len(ex), 32):
        chunk = [ds[ex[j].image_index][0] for j in range(i, min(i + 32, len(ex)))]
        img_emb.append(sig.image_embeds(chunk))
    img_emb = torch.cat(img_emb)
    print(f"  siglip: {len(img_emb)} images in {time.time()-t0:.0f}s")
    del sig
    torch.mps.empty_cache() if cfg.device == "mps" else None

    # ---- arm A: cross-encoder through the VLM with the block mask
    from vision.features import FrozenEncoder
    enc = FrozenEncoder(cfg, mask_mode=args.mask)
    d = enc.hidden_size
    slots = np.zeros((len(ex), K, d), dtype=np.float16)
    smask = np.zeros((len(ex), K), dtype=bool)
    t0 = time.time()
    for i, e in enumerate(ex):
        img = ds[e.image_index][0]
        packed = pack_one(enc.processor, img, e.question, e.options,
                          max_option_tokens=cfg.max_option_tokens)
        s, m, _ = enc.slot_states([packed])
        slots[i, :s.shape[1]] = s[0].numpy().astype(np.float16)
        smask[i, :m.shape[1]] = m[0].numpy()
        if i % 100 == 0 and i:
            r = (time.time() - t0) / i
            print(f"  vlm {i}/{len(ex)}  {r*1000:.0f} ms/ex  "
                  f"eta {r*(len(ex)-i)/60:.1f} min", flush=True)
    print(f"  vlm: {len(ex)} forwards in {(time.time()-t0)/60:.1f} min")

    np.savez_compressed(
        f"{OUT}/{args.split}.npz",
        slots=slots, slot_mask=smask,
        answer=np.array([e.answer for e in ex]),
        mode=np.array([e.mode for e in ex]),
        k=np.array([e.k for e in ex]),
        img_emb=img_emb.numpy().astype(np.float32),
        lab_emb=lab_emb.numpy().astype(np.float32),
        opt_ix=np.array([[lab_ix[o] for o in e.options] + [-1] * (K - e.k)
                         for e in ex]),
        is_none=is_none_matrix([e.options for e in ex], K).numpy(),
    )
    print(f"  -> {OUT}/{args.split}.npz")


def _load(split):
    z = np.load(f"{OUT}/{split}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def _split(d, frac=0.8, seed=0):
    """Hold out a validation slice of TRAINVAL for early stopping.

    The test split is never touched during fitting. Selecting on test would
    quietly inflate every number in the table, and calibration numbers are
    especially easy to flatter that way.
    """
    n = len(d["answer"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(n * frac)
    take = lambda ix: {k: (v[ix] if getattr(v, "shape", None) and
                           v.shape[:1] == (n,) else v) for k, v in d.items()}
    return take(perm[:cut]), take(perm[cut:])


def _targets(d):
    y = torch.from_numpy(d["answer"]).long()
    K = d["slot_mask"].shape[1]
    tgt = torch.zeros(len(y), K)
    tgt[torch.arange(len(y)), y] = 1.0
    return y, tgt, torch.from_numpy(d["slot_mask"])


def _nll(logits, d):
    y, _, sm = _targets(d)
    lp = torch.log_softmax(logits.masked_fill(~sm, -1e9), -1)
    return -lp[torch.arange(len(y)), y].mean().item()


def _fit(head, forward, tr, va, steps=600, lr=3e-3):
    """Full-batch AdamW. The features are frozen, so this is a convex-ish fit
    on a few thousand rows and takes seconds."""
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    _, tgt, sm = _targets(tr)
    best, best_state = 1e9, {k: v.clone() for k, v in head.state_dict().items()}
    for s in range(steps):
        head.train()
        loss, _ = decision_loss(forward(head, tr), tgt, sm)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 20 == 0 or s == steps - 1:
            head.eval()
            with torch.no_grad():
                vl = _nll(forward(head, va), va)
            if vl < best:
                best = vl
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, best


class CrossHead(torch.nn.Module):
    """The shared d->1 probe, applied to already-gathered option states.

    Identical in substance to systemone.model.head.SlotHead; it just skips the
    gather, because caching the frozen backbone already did it.
    """

    def __init__(self, d):
        super().__init__()
        self.probe = torch.nn.Linear(d, 1)
        torch.nn.init.normal_(self.probe.weight, std=0.02)
        torch.nn.init.zeros_(self.probe.bias)

    def forward(self, H, slot_mask):
        logits = self.probe(H).squeeze(-1)
        return logits.masked_fill(~slot_mask, -1e4)


def train(args):
    full, test = _load("trainval"), _load("test")
    tr, va = _split(full)
    d = tr["slots"].shape[-1]
    print(f"train {len(tr['answer'])}  val {len(va['answer'])}  "
          f"test {len(test['answer'])}  d={d}")
    results = {}

    def cross_fwd(h, dd):
        return h(torch.from_numpy(dd["slots"]).float(),
                 torch.from_numpy(dd["slot_mask"]))

    probe, vl = _fit(CrossHead(d), cross_fwd, tr, va)
    with torch.no_grad():
        results["cross-encoder probe"] = (
            cross_fwd(probe, test), sum(p.numel() for p in probe.parameters()))
    print(f"  cross-encoder probe   val nll {vl:.4f}")

    cos = {k: _bi_cos(v) for k, v in (("tr", tr), ("va", va), ("te", test))}
    key = lambda dd: "tr" if dd is tr else ("va" if dd is va else "te")

    h1, vl1 = _fit(BiEncoderHead(),
                   lambda h, dd: h(cos[key(dd)][0],
                                   torch.from_numpy(dd["slot_mask"]),
                                   torch.from_numpy(dd["is_none"])), tr, va)
    with torch.no_grad():
        results["bi-encoder (temp)"] = (
            h1(cos["te"][0], torch.from_numpy(test["slot_mask"]),
               torch.from_numpy(test["is_none"])),
            sum(p.numel() for p in h1.parameters()))
    print(f"  bi-encoder temp       val nll {vl1:.4f}")

    dim = cos["tr"][1].shape[-1]
    h2, vl2 = _fit(DiagHead(dim),
                   lambda h, dd: h(cos[key(dd)][1], cos[key(dd)][2],
                                   torch.from_numpy(dd["slot_mask"]),
                                   torch.from_numpy(dd["is_none"])), tr, va)
    with torch.no_grad():
        results["bi-encoder (diag)"] = (
            h2(cos["te"][1], cos["te"][2], torch.from_numpy(test["slot_mask"]),
               torch.from_numpy(test["is_none"])),
            sum(p.numel() for p in h2.parameters()))
    print(f"  bi-encoder diag       val nll {vl2:.4f}")

    # what people actually ship: raw cosine, temperature 100, no training
    results["bi-encoder (zero-shot)"] = (cos["te"][0] * 100.0, 0)

    torch.save(results, f"{OUT}/logits.pt")
    report(args)


def report(args):
    va = _load("test")
    res = torch.load(f"{OUT}/logits.pt", weights_only=False)
    y = va["answer"]; sm = torch.from_numpy(va["slot_mask"]); mode = va["mode"]

    rows = []
    for name, (logits, nparam) in res.items():
        p = torch.softmax(logits.masked_fill(~sm, -1e9), -1).numpy()
        pred = p.argmax(1); correct = (pred == y).astype(float)
        ptop = p.max(1)
        bd = brier_decomposition(ptop, correct)
        row = dict(name=name, params=nparam, acc=correct.mean(),
                   ece=ece(ptop, correct), brier=bd["brier"],
                   resolution=bd["resolution"])
        for m in ["random", "confusable", "abstain"]:
            sel = mode == m
            row[m] = correct[sel].mean() if sel.sum() else float("nan")
        rows.append(row)

    print(f"\n{'arm':<26}{'params':>9}{'acc':>8}{'ECE':>8}{'Brier':>8}"
          f"{'resol':>8}{'random':>9}{'confus':>9}{'abstain':>9}")
    print("-" * 94)
    for r in rows:
        print(f"{r['name']:<26}{r['params']:>9,}{r['acc']:>8.3f}{r['ece']:>8.3f}"
              f"{r['brier']:>8.3f}{r['resolution']:>8.3f}{r['random']:>9.3f}"
              f"{r['confusable']:>9.3f}{r['abstain']:>9.3f}")
    json.dump(rows, open(f"{OUT}/report.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache"); c.add_argument("--split", default="trainval")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--mask", default="block", choices=["block", "causal"])
    sub.add_parser("train"); sub.add_parser("report")
    a = ap.parse_args()
    {"cache": cache, "train": train, "report": report}[a.cmd](a)
