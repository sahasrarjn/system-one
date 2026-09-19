"""Produce the deliverable: the reliability diagram, plus the three numbers
that matter more than accuracy."""
import argparse, json
import numpy as np

from systemone.evaluation.metrics import (
    reliability, ece, brier_decomposition, selective, coverage_at)


def ascii_reliability(pts, width=46):
    """Terminal-native, so it works over SSH on a headless box."""
    print("\n  stated -> observed        (· perfect, # actual)")
    for conf, acc, n in pts:
        row = [" "] * width
        row[min(width - 1, int(conf * (width - 1)))] = "."
        row[min(width - 1, int(acc * (width - 1)))] = "#"
        print(f"  {conf:.2f} {''.join(row)} {acc:.2f}  n={n}")
    print(f"  {'0.0':<6}{'':<{width-8}}{'1.0'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="artifacts/runs/run1/val_preds.npz")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--plot", help="optional PNG path (needs matplotlib)")
    args = ap.parse_args()

    d = np.load(args.preds)
    p, c, k = d["p_top"], d["correct"], d["conf"]

    pts = reliability(p, c, args.bins)
    dec = brier_decomposition(p, c, args.bins)
    out = {
        "n": int(len(p)),
        "accuracy": float(c.mean()),
        "ece_equal_mass": ece(p, c, args.bins),
        **dec,
        "coverage_at_90pct_acc": coverage_at(k, c, 0.90),
        "coverage_at_95pct_acc": coverage_at(k, c, 0.95),
    }
    ascii_reliability(pts)
    print()
    for key, v in out.items():
        print(f"  {key:24s} {v:.4f}" if isinstance(v, float) else f"  {key:24s} {v}")
    print("\n  resolution is the one to watch: a base-rate guesser is")
    print("  perfectly calibrated and perfectly useless.\n")

    if args.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        ax[0].plot([0, 1], [0, 1], "--", lw=1, color="#888", label="perfect")
        ax[0].plot([x for x, _, _ in pts], [y for _, y, _ in pts],
                   "o-", color="#8C2F39", label="model")
        ax[0].set(xlabel="stated probability", ylabel="observed frequency",
                  title=f"Reliability (ECE={out['ece_equal_mass']:.3f})")
        ax[0].legend()
        cov, acc = selective(k, c)
        ax[1].plot(cov, acc, color="#0F6E5C")
        ax[1].set(xlabel="coverage", ylabel="accuracy",
                  title="Selective prediction")
        fig.tight_layout(); fig.savefig(args.plot, dpi=140)
        print(f"  wrote {args.plot}")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
