"""Step 5: Partial reproduction of Table 4 - quantitative XAI metrics.

Computes Detection (Det) and Deletion (Del) for CLIP-QDA_local, CLIP-LIME and
CLIP-SHAP on the Cats/Dogs/Cars dataset.

Det (Detection, higher is better) - reproduced per the paper's protocol.
  The paper uses Ss=[Black, White] on biased binary classifiers and averages
  over all such tasks. We do the same: all 6 biased binary tasks
  (Black X + White Y, X!=Y), a fresh QDA per task, and we check whether the
  bias concepts appear in the explanation's top-2.
  Paper Det: local=0.2724  lime=0.4042  shap=0.3696

Del (Deletion, lower is better) - internal sanity check only, NOT comparable
  to the paper. We report Del with a random concept ordering vs the method's
  importance ordering; the method ordering should give the lower Del. The
  paper's Del Set 1 / Set 2 instead denote two concept vocabularies (the real
  Table-6 concepts vs an equal number of random dictionary words), evaluated
  with each method's ordering - a setup we do not reproduce.
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from clip_scores import load_scores, scores_matrix          # noqa: E402
from concepts import get_concept_set                         # noqa: E402
from qda_model import CLIPQDA, train_eval                    # noqa: E402
from utils import ensure_dir, load_config, save_json, set_seed  # noqa: E402
from xai import (                                            # noqa: E402
    deletion_metric, detection_metric,
    importance_local, importance_lime, importance_shap,
)

N_MAX = 9        # paper default
LIME_SAMPLES = 500
SHAP_NSAMPLES = 100

# Helpers

def _make_importance_fn(method, concepts, class_names, X_train, seed):
    """Return a closure (z, clf, X_train) -> list[int] for the given method."""
    if method == "local":
        return lambda z, clf, Xtr: importance_local(z, clf, Xtr, concepts)
    if method == "lime":
        return lambda z, clf, Xtr: importance_lime(
            z, clf, Xtr, concepts, class_names,
            num_samples=LIME_SAMPLES, seed=seed)
    if method == "shap":
        return lambda z, clf, Xtr: importance_shap(
            z, clf, Xtr, concepts, nsamples=SHAP_NSAMPLES, seed=seed)
    raise ValueError(method)


def _del_random_seed(seed, method):
    """Different seeds per method so random baselines are independent."""
    return seed + hash(method) % 1000



def compute_deletion(scores_df, concept_cols, concepts, class_names,
                     methods, seed,
                     n_max=N_MAX, lime_samples=LIME_SAMPLES,
                     shap_nsamples=SHAP_NSAMPLES):
    species = ["Cat", "Dog", "Car"]
    colors  = ["Black", "White"]
    biased_tasks = [
        (c1, s1, c2, s2)
        for (c1, s1), (c2, s2) in itertools.permutations(
            itertools.product(colors, species), 2)
        if s1 != s2 and c1 != c2 and c1 == "Black" and c2 == "White"
    ]

    # Per-method: accumulate Del values over tasks; track timing on first task.
    del1_acc  = {m: [] for m in methods}
    del2_acc  = {m: [] for m in methods}
    times     = {m: 0.0 for m in methods}
    n_test_total = {m: 0 for m in methods}

    for c1, s1, c2, s2 in biased_tasks:
        task_label = f"Black {s1}s vs White {s2}s"
        td = _biased_task_data(scores_df, c1, s1, c2, s2, concept_cols, seed)
        if td is None:
            continue
        X_tr, y_tr, X_te, y_te = td
        try:
            clf_b, _ = train_eval(X_tr, y_tr, X_te, y_te, reg_param=1e-4)
        except Exception as exc:  # noqa: BLE001
            print(f"  [del] skip {task_label}: {exc}")
            continue

        task_classes = list(clf_b.classes_)
        print(f"\n[del] {task_label} | test={len(X_te)}")

        for method in methods:
            imp_fn = _make_importance_fn(method, concepts, task_classes,
                                         X_tr, seed)
            t0 = time.time()
            d2, _ = deletion_metric(X_te, y_te, clf_b, imp_fn, X_tr,
                                    n_max=n_max)
            times[method] += time.time() - t0
            n_test_total[method] += len(X_te)

            d1, _ = deletion_metric(X_te, y_te, clf_b, imp_fn, X_tr,
                                    n_max=n_max,
                                    random_seed=_del_random_seed(seed, method))
            del1_acc[method].append(d1)
            del2_acc[method].append(d2)
            print(f"    {method}: Del1={d1:.4f}  Del2={d2:.4f}")

    results = {}
    for method in methods:
        d1 = float(np.mean(del1_acc[method])) if del1_acc[method] else float("nan")
        d2 = float(np.mean(del2_acc[method])) if del2_acc[method] else float("nan")
        results[method] = {
            "del_set1": round(d1, 4),
            "del_set2": round(d2, 4),
            "inference_time_s": round(times[method], 2),
            "n_test": n_test_total[method],
        }
        print(f"\n  {method}: Del(Set1)={d1:.4f}  Del(Set2)={d2:.4f}")
    return results


# Detection metric (Det): biased binary tasks as in the paper

BIAS_CONCEPTS = ["Black", "White"]   # Ss for every sample in a biased task


def _biased_task_data(scores_df, color1, species1, color2, species2,
                      concept_cols, seed):
    """Build (X_train, y_train, X_test, y_test) for one biased binary task."""
    from utils import stratified_split

    label1 = f"{color1} {species1}s"   # e.g. "Black Cats"
    label2 = f"{color2} {species2}s"

    mask = scores_df["raw_label"].isin([label1, label2])
    sub = scores_df[mask].reset_index(drop=True)
    if len(sub) < 4:
        return None

    # Binary label: species name (Cat/Dog/Car) so QDA class names are species.
    sub = sub.copy()
    sub["bin_label"] = sub["label"]   # already Cat/Dog/Car

    # Deterministic 80/20 split within this subset.
    split = stratified_split(sub["bin_label"].values, 0.2, seed)
    sub["split"] = split

    X = sub[concept_cols].to_numpy(dtype=np.float64)
    y = sub["bin_label"].to_numpy()
    tr = sub["split"] == "train"
    te = sub["split"] == "test"
    if tr.sum() < 2 or te.sum() < 2:
        return None
    return X[tr], y[tr], X[te], y[te]


def compute_detection(scores_df, concept_cols, concepts, class_names,
                      methods, seed, lime_samples=LIME_SAMPLES,
                      shap_nsamples=SHAP_NSAMPLES):
    """Average Det over all 6 biased binary tasks."""
    species = ["Cat", "Dog", "Car"]
    colors = ["Black", "White"]

    # All (color1, species1, color2, species2) pairs where species differ
    # and one is Black, the other White (the biased scenario).
    biased_tasks = [
        (c1, s1, c2, s2)
        for (c1, s1), (c2, s2) in itertools.permutations(
            itertools.product(colors, species), 2
        )
        if s1 != s2 and c1 != c2
    ]
    # Deduplicate unordered pairs: keep only (Black X, White Y).
    biased_tasks = [
        (c1, s1, c2, s2) for c1, s1, c2, s2 in biased_tasks
        if c1 == "Black" and c2 == "White"
    ]

    # Concept indices for "Black" and "White".
    bias_idx = set(i for i, c in enumerate(concepts) if c in BIAS_CONCEPTS)

    def ground_truth_fn(_pred):
        return bias_idx   # Ss = {Black, White} for all samples

    results = {m: [] for m in methods}
    for c1, s1, c2, s2 in biased_tasks:
        task_label = f"Black {s1}s vs White {s2}s"
        td = _biased_task_data(scores_df, c1, s1, c2, s2, concept_cols, seed)
        if td is None:
            print(f"  [det] skip {task_label} (too few samples)")
            continue
        X_tr, y_tr, X_te, y_te = td

        # Train fresh QDA on this biased subset.
        try:
            clf_b, _ = train_eval(X_tr, y_tr, X_te, y_te,
                                  reg_param=1e-4)
        except Exception as exc:  # noqa: BLE001
            print(f"  [det] skip {task_label}: {exc}")
            continue

        # Subset of concepts relevant to this task.
        task_classes = list(clf_b.classes_)

        print(f"  [det] {task_label} | test={len(X_te)}")
        for method in methods:
            imp_fn = _make_importance_fn(method, concepts, task_classes, X_tr,
                                         seed)
            det = detection_metric(X_te, clf_b, imp_fn, X_tr,
                                   ground_truth_fn, top_k=len(bias_idx))
            results[method].append(det)
            print(f"    {method}: Det={det:.4f}")

    # Average over tasks.
    return {m: round(float(np.mean(v)), 4) if v else float("nan")
            for m, v in results.items()}


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=os.path.join(ROOT, "configs", "cats_dogs_cars.yaml"))
    ap.add_argument("--methods", nargs="*",
                    default=["local", "lime", "shap"],
                    help="Subset of: local lime shap")
    ap.add_argument("--n-max", type=int, default=N_MAX)
    ap.add_argument("--lime-samples", type=int, default=LIME_SAMPLES)
    ap.add_argument("--shap-nsamples", type=int, default=SHAP_NSAMPLES)
    args = ap.parse_args()

    # Allow CLI overrides of the module-level defaults.
    n_max = args.n_max
    lime_samples = args.lime_samples
    shap_nsamples = args.shap_nsamples

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    scores_df = load_scores(os.path.join(ROOT, cfg["scores_csv"]))
    # raw_label is needed for biased task detection.
    if "raw_label" not in scores_df.columns:
        raise RuntimeError(
            "scores CSV missing 'raw_label' column. Re-run script 01.")
    data = scores_matrix(scores_df)
    concept_cols = data["concept_cols"]
    concepts = [c.replace("concept::", "") for c in concept_cols]
    info = get_concept_set(cfg["dataset"])
    class_names = info["class_names"]
    methods = args.methods

    # Load or retrain QDA.
    model_path = os.path.join(ROOT, cfg["model_path"])
    if os.path.exists(model_path):
        clf = CLIPQDA.load(model_path)
    else:
        clf, _ = train_eval(data["X_train"], data["y_train"],
                            data["X_test"], data["y_test"],
                            reg_param=cfg.get("reg_param", 1e-4))

    print("=" * 60)
    print("Table 4 partial reproduction (Cats/Dogs/Cars)")
    print(f"methods={methods}  n_max={n_max}  "
          f"n_test={len(data['X_test'])}")
    print("=" * 60)

    # ---- Deletion ---------------------------------------------------------- #
    print("\n--- Deletion metric (biased binary tasks) ---")
    del_results = compute_deletion(scores_df, concept_cols, concepts,
                                   class_names, methods, seed,
                                   n_max=n_max,
                                   lime_samples=lime_samples,
                                   shap_nsamples=shap_nsamples)

    # ---- Detection --------------------------------------------------------- #
    print("\n--- Detection metric (biased binary tasks) ---")
    det_results = compute_detection(
        scores_df, concept_cols, concepts, class_names, methods, seed,
        lime_samples=lime_samples, shap_nsamples=shap_nsamples)

    paper_det = {"local": 0.2724, "lime": 0.4042, "shap": 0.3696}

    rows = []
    for m in methods:
        d = del_results[m]
        det = det_results.get(m, float("nan"))
        rows.append({
            "method": "CLIP-QDA_local" if m == "local" else "CLIP-" + m.upper(),
            "del_random_order": d["del_set1"],
            "del_method_order": d["del_set2"],
            "det_ours": det,
            "det_paper": paper_det.get(m, ""),
            "inference_time_s": d["inference_time_s"],
            "n_test": d["n_test"],
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, "results", "table4_xai_metrics.csv")
    out_json = os.path.join(ROOT, "results", "table4_xai_metrics.json")
    ensure_dir(out_csv)
    df.to_csv(out_csv, index=False)
    save_json({
        "deletion": del_results,
        "detection": det_results,
        "table": rows,
        "note": (
            "Det is computed per the paper's protocol: averaged over the 6 biased"
            " binary tasks (Black X + White Y, X!=Y) with ground-truth Ss={Black,"
            " White}, checking whether the bias concepts appear in the top-2."
            " Del is an internal sanity check only (method order vs random order);"
            " it is NOT comparable to the paper's Del Set 1 / Set 2, which are two"
            " concept vocabularies (real vs random words), not orderings."
        ),
    }, out_json)

    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
