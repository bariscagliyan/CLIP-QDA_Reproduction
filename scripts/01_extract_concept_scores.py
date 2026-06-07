"""Step 1: compute and cache CLIP concept scores to CSV.

The cache lets the QDA classifier and the XAI experiments be re-run many times
without paying the cost of the CLIP forward passes again.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from clip_scores import extract_scores, save_scores  # noqa: E402
from data import load_dataset_by_name  # noqa: E402
from utils import get_device, load_config, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a YAML config.")
    ap.add_argument("--clip-model", default=None,
                    help="Override CLIP model (e.g. ViT-B/32 for fast debug).")
    ap.add_argument("--data-root", default=None,
                    help="Override MonuMAI data root.")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Override max samples (Cats/Dogs/Cars).")
    ap.add_argument("--out", default=None, help="Override output CSV path.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.clip_model:
        cfg["clip_model"] = args.clip_model
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.max_samples is not None:
        cfg["max_samples"] = args.max_samples

    set_seed(cfg.get("seed", 42))
    device = get_device()
    print("[01] device:", device, "| CLIP:", cfg.get("clip_model"))

    df, get_image = load_dataset_by_name(cfg)
    print("[01] loaded {} samples ({} split)".format(
        len(df), df.attrs.get("split_origin", "?")))

    scores = extract_scores(df, get_image, cfg, device=device)

    out = args.out or os.path.join(ROOT, cfg["scores_csv"])
    save_scores(scores, out)


if __name__ == "__main__":
    main()
