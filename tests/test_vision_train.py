"""Run the vision comparison end to end on synthetic cached features.

This exists because `_bi_cos` went missing in an edit and the NameError only
surfaced on a paid GPU instance, three stages and six minutes in, on a code
path no local test ever touched. The features are frozen vectors by the time
`train` sees them, so nothing here needs a model, a GPU, or a dataset.
"""
import os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_TR, N_TE, K, D, DIM, N_LAB = 240, 120, 6, 32, 16, 12


def fake_split(n, rng):
    slots = rng.normal(size=(n, K, D)).astype(np.float16)
    slot_mask = np.zeros((n, K), dtype=bool)
    ks = rng.integers(2, K + 1, size=n)
    for i, k in enumerate(ks):
        slot_mask[i, :k] = True
    answer = np.array([rng.integers(0, k) for k in ks])
    img = rng.normal(size=(n, DIM)).astype(np.float32)
    img /= np.linalg.norm(img, axis=-1, keepdims=True)
    lab = rng.normal(size=(N_LAB, DIM)).astype(np.float32)
    lab /= np.linalg.norm(lab, axis=-1, keepdims=True)
    opt_ix = rng.integers(0, N_LAB, size=(n, K))
    is_none = np.zeros((n, K), dtype=bool)
    for i, k in enumerate(ks):           # last real slot is the escape hatch
        if rng.random() < 0.5:
            is_none[i, k - 1] = True
    mode = np.array(["confusable" if i % 2 else "random" for i in range(n)])
    mode[::7] = "abstain"
    return dict(slots=slots, slot_mask=slot_mask, answer=answer, mode=mode,
                k=ks, img_emb=img, lab_emb=lab, opt_ix=opt_ix, is_none=is_none)


def main():
    import scripts.vision_run as vr
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        vr.OUT = d
        np.savez_compressed(f"{d}/trainval.npz", **fake_split(N_TR, rng))
        np.savez_compressed(f"{d}/test.npz", **fake_split(N_TE, rng))

        class A: pass
        vr.train(A())                      # trains all arms, then reports

        res = __import__("torch").load(f"{d}/logits.pt", weights_only=False)
        print("\narms produced:", len(res))
        for name, (logits, nparam) in res.items():
            assert logits.shape == (N_TE, K), f"{name}: {logits.shape}"
            assert np.isfinite(logits.numpy()).all(), f"{name}: non-finite logits"
            print(f"  {name:<26} {nparam:>8,} params  logits {tuple(logits.shape)}")
        assert len(res) == 4, f"expected 4 arms, got {len(res)}"
        import json
        rows = json.load(open(f"{d}/report.json"))
        assert len(rows) == 4
        print("\nall four arms train, produce finite logits, and report.")


if __name__ == "__main__":
    main()
