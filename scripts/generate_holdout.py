"""
Builds a holdout file for the 'Test on New Data' tab.

The dashboard trains on a random sample drawn with a fixed seed. This
script reproduces that sample, removes it, and draws the holdout from what
is left, so the rows were never available to the models under any split.

Usage:
    python scripts/generate_holdout.py                 # 5,000 rows
    python scripts/generate_holdout.py --rows 20000
    python scripts/generate_holdout.py --no-labels     # drop is_fraud
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "nibss_fraud_dataset.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5_000,
                    help="Number of holdout transactions to write.")
    ap.add_argument("--trained-on", type=int, default=200_000,
                    help="Sample size the dashboard was trained with "
                         "(must match the sidebar setting).")
    ap.add_argument("--no-labels", action="store_true",
                    help="Drop is_fraud and fraud_technique to simulate live "
                         "traffic with no ground truth.")
    ap.add_argument("--seed", type=int, default=7,
                    help="Seed for the holdout draw. Kept different from the "
                         "training seed of 42.")
    ap.add_argument("--out", type=str, default="holdout_sample.csv")
    args = ap.parse_args()

    print(f"Reading {SOURCE.name}...")
    df = pd.read_csv(SOURCE, low_memory=False)

    # Reproduce the training sample (seed 42, as in load_dataset) so it can
    # be excluded.
    if args.trained_on < len(df):
        trained_idx = df.sample(n=args.trained_on, random_state=42).index
    else:
        raise SystemExit(
            "Training sample covers the whole dataset, so no unseen rows remain. "
            "Lower --trained-on to match a smaller sidebar sample size.")

    unseen = df.drop(index=trained_idx)
    print(f"Rows never seen in training: {len(unseen):,}")

    n = min(args.rows, len(unseen))
    holdout = unseen.sample(n=n, random_state=args.seed)

    if args.no_labels:
        holdout = holdout.drop(columns=["is_fraud", "fraud_technique"],
                               errors="ignore")
        note = "labels stripped (blind test)"
    else:
        note = f"{int(holdout['is_fraud'].sum())} fraud cases included"

    out_path = ROOT / "samples" / args.out
    holdout.to_csv(out_path, index=False)
    print(f"Wrote {n:,} unseen transactions to samples/{out_path.name} ({note})")


if __name__ == "__main__":
    main()
