"""Step 0: obtain the datasets.

- MonuMAI: tries to clone ari-dasci/OD-MonuMAI (or uses --data-root).
- Cats/Dogs/Cars: checks that the Hugging Face dataset is reachable.

This script only downloads / verifies. CLIP scoring happens in step 01.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils import load_config  # noqa: E402


def prepare_monumai(args):
    from data import load_monumai
    print("\n=== MonuMAI ===")
    try:
        df, _ = load_monumai(data_root=args.data_root,
                             clone_dir=os.path.join(ROOT, "data", "OD-MonuMAI"))
    except Exception as exc:  # noqa: BLE001
        print("[!] MonuMAI not ready:", exc)
        print("    Download manually and re-run with --data-root <path>.")
        return
    print("Samples:", len(df))
    print("Split origin:", df.attrs.get("split_origin"))
    print(df.groupby(["label", "split"]).size())


def prepare_cdc(args):
    print("\n=== Cats/Dogs/Cars ===")
    cfg = load_config(os.path.join(ROOT, "configs", "cats_dogs_cars.yaml"))
    try:
        from data import load_cats_dogs_cars
        df, _ = load_cats_dogs_cars(
            test_size=cfg["test_size"],
            seed=cfg["seed"],
            max_samples=args.max_samples or cfg.get("max_samples"),
        )
    except Exception as exc:  # noqa: BLE001
        print("[!] Could not load HF dataset:", exc)
        print("    Check your internet connection / `datasets` install.")
        return
    print("Samples:", len(df))
    print(df.groupby(["label", "split"]).size())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["monumai", "cats_dogs_cars", "both"],
                    default="both")
    ap.add_argument("--data-root", default=None,
                    help="Local MonuMAI path (used if cloning fails).")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Optional cap on Cats/Dogs/Cars sample count.")
    args = ap.parse_args()

    if args.dataset in ("monumai", "both"):
        prepare_monumai(args)
    if args.dataset in ("cats_dogs_cars", "both"):
        prepare_cdc(args)


if __name__ == "__main__":
    main()
