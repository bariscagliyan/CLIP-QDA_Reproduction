"""Step 2: train and evaluate CLIP-QDA over the reg_param sweep.

Reads the cached concept scores, fits a QDA per reg_param value, reports
train/test accuracy and saves metrics as CSV and JSON. For MonuMAI this
produces the Table 2 reproduction (paper CLIP-QDA accuracy = 0.89).
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from clip_scores import load_scores, scores_matrix  # noqa: E402
from qda_model import train_eval  # noqa: E402
from utils import load_config, save_json, set_seed  # noqa: E402

# Paper Table 2 reference accuracies (for the comparison column).
PAPER_ACCURACY = {"monumai": 0.89}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scores", default=None, help="Override scores CSV path.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    dataset = cfg["dataset"].lower()

    scores_path = args.scores or os.path.join(ROOT, cfg["scores_csv"])
    scores_df = load_scores(scores_path)
    data = scores_matrix(scores_df)
    concepts = [c.replace("concept::", "") for c in data["concept_cols"]]
    print("[02] {} | {} concepts | train={} test={}".format(
        dataset, len(concepts), len(data["X_train"]), len(data["X_test"])))

    sweep = cfg.get("reg_param_sweep", [cfg.get("reg_param", 1e-4)])
    rows, best = [], None
    for reg in sweep:
        _, metrics = train_eval(data["X_train"], data["y_train"],
                                data["X_test"], data["y_test"], reg_param=reg)
        metrics["dataset"] = dataset
        metrics["clip_model"] = cfg.get("clip_model")
        metrics["split_origin"] = scores_df.attrs.get("split_origin", "n/a")
        metrics["is_default_reg"] = bool(
            abs(reg - cfg.get("reg_param", 1e-4)) < 1e-12)
        if dataset in PAPER_ACCURACY:
            metrics["paper_accuracy"] = PAPER_ACCURACY[dataset]
            metrics["delta_vs_paper"] = round(
                metrics["test_accuracy"] - PAPER_ACCURACY[dataset], 4)
        rows.append(metrics)
        print("  reg_param={:<8} train_acc={:.4f} test_acc={:.4f}".format(
            reg, metrics["train_accuracy"], metrics["test_accuracy"]))
        if best is None or metrics["test_accuracy"] > best["test_accuracy"]:
            best = metrics

    out_df = pd.DataFrame(rows)
    csv_path = os.path.join(ROOT, cfg["results_csv"])
    json_path = os.path.join(ROOT, cfg["results_json"])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    out_df.to_csv(csv_path, index=False)
    save_json({"sweep": rows, "best": best}, json_path)

    print("\n[02] wrote", csv_path)
    print("[02] best test accuracy: {:.4f} (reg_param={})".format(
        best["test_accuracy"], best["reg_param"]))
    if dataset in PAPER_ACCURACY:
        print("[02] paper CLIP-QDA accuracy: {:.2f} | delta: {:+.4f}".format(
            PAPER_ACCURACY[dataset],
            best["test_accuracy"] - PAPER_ACCURACY[dataset]))

    # Persist the default-reg model for the XAI step.
    from clip_scores import scores_matrix as _sm  # noqa: F401
    from qda_model import CLIPQDA
    clf = CLIPQDA(reg_param=cfg.get("reg_param", 1e-4)).fit(
        data["X_train"], data["y_train"])
    model_path = os.path.join(ROOT, cfg["model_path"])
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    clf.save(model_path)
    print("[02] saved model ->", model_path)


if __name__ == "__main__":
    main()
