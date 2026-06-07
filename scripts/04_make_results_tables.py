"""Step 4: assemble the final result tables and print a summary.

Reads the metric files produced by step 02 for each dataset and prints a
compact summary, including the Table 2 comparison for MonuMAI.
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils import load_config  # noqa: E402


def summarise(config_path):
    cfg = load_config(config_path)
    csv_path = os.path.join(ROOT, cfg["results_csv"])
    if not os.path.exists(csv_path):
        print("[04] missing:", csv_path, "(run step 02 first)")
        return None
    df = pd.read_csv(csv_path)
    print("\n=== {} ===".format(cfg["dataset"]))
    cols = [c for c in ("reg_param", "train_accuracy", "test_accuracy",
                        "paper_accuracy", "delta_vs_paper") if c in df.columns]
    print(df[cols].to_string(index=False))
    best = df.loc[df["test_accuracy"].idxmax()]
    print("-> best test accuracy: {:.4f} (reg_param={})".format(
        best["test_accuracy"], best["reg_param"]))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=[
        os.path.join(ROOT, "configs", "monumai.yaml"),
        os.path.join(ROOT, "configs", "cats_dogs_cars.yaml"),
    ])
    args = ap.parse_args()

    print("=" * 60)
    print("CLIP-QDA reproduction summary")
    print("=" * 60)
    for cfg_path in args.configs:
        if os.path.exists(cfg_path):
            summarise(cfg_path)

    print("\nArtifacts:")
    for rel in ("results/table2_monumai_reproduction.csv",
                "results/cats_dogs_cars_accuracy.csv",
                "results/concept_scores/monumai_scores.csv",
                "results/concept_scores/cats_dogs_cars_scores.csv"):
        path = os.path.join(ROOT, rel)
        print("  [{}] {}".format("x" if os.path.exists(path) else " ", rel))


if __name__ == "__main__":
    main()
