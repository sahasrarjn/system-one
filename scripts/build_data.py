"""Build the training set. No GPU, no teacher, no annotation budget."""
import argparse, random, os

from systemone.config import Config
from systemone.data import civil, clinc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinc-n", type=int, default=20_000)
    ap.add_argument("--civil-n", type=int, default=40_000)
    ap.add_argument("--out-dir", default="artifacts/data")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = []

    if args.clinc_n:
        print("loading CLINC150 (150 intents + out-of-scope)...")
        records += list(clinc.load(args.clinc_n))
        print(f"  {len(records):,} utterances")

    print("loading Civil Comments (human annotator fractions)...")
    before = len(records)
    records += list(civil.load(args.civil_n))
    print(f"  {len(records)-before:,} comments")

    random.seed(0)
    random.shuffle(records)
    n_val = int(len(records) * args.val_frac)

    for name, subset in [("val", records[:n_val]), ("train", records[n_val:])]:
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in subset:
                f.write(r.to_json() + "\n")
        nq = sum(len(r.questions) for r in subset)
        print(f"{path}: {len(subset):,} records / {nq:,} questions")


if __name__ == "__main__":
    main()
