"""Step 3: limited XAI experiments on Cats/Dogs/Cars.

Runs the four explanation families on a handful of test samples and saves:
  * a JSON file with per-sample local / LIME / SHAP explanations,
  * a global signed-Wasserstein ranking per class pair,
  * one bar-plot figure per explained sample.

Designed to be light (a few samples) so it runs quickly on a laptop.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from clip_scores import load_scores, scores_matrix  # noqa: E402
from concepts import get_concept_set  # noqa: E402
from qda_model import CLIPQDA, train_eval  # noqa: E402
from utils import ensure_dir, load_config, save_json, set_seed  # noqa: E402
from xai import (clip_lime, clip_shap, global_explanation,  # noqa: E402
                 local_counterfactuals)


def barplot(pairs, title, path, xlabel="value"):
    labels = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in values]
    plt.figure(figsize=(7, 0.5 * len(labels) + 1.5))
    plt.barh(range(len(labels)), values, color=colors)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    ensure_dir(path)
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=os.path.join(ROOT, "configs", "cats_dogs_cars.yaml"))
    ap.add_argument("--num-samples", type=int, default=4,
                    help="Number of test images to explain.")
    ap.add_argument("--lime-samples", type=int, default=1000)
    ap.add_argument("--shap-nsamples", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    scores_df = load_scores(os.path.join(ROOT, cfg["scores_csv"]))
    data = scores_matrix(scores_df)
    info = get_concept_set(cfg["dataset"])
    concepts = info["concepts"]
    class_names = info["class_names"]

    model_path = os.path.join(ROOT, cfg["model_path"])
    if os.path.exists(model_path):
        clf = CLIPQDA.load(model_path)
    else:
        clf, _ = train_eval(data["X_train"], data["y_train"],
                            data["X_test"], data["y_test"],
                            reg_param=cfg.get("reg_param", 1e-4))

    xai_dir = os.path.join(ROOT, cfg["xai_dir"])
    ensure_dir(xai_dir)

    # Global explanations (one ranking per ordered class pair) 
    global_out = {}
    for a in range(len(class_names)):
        for b in range(len(class_names)):
            if a == b:
                continue
            c1, c2 = class_names[a], class_names[b]
            ranking = global_explanation(clf, c1, c2, concepts, top_k=len(concepts))
            global_out["{} vs {}".format(c1, c2)] = ranking
            if b == (a + 1) % len(class_names):  # one figure per class
                barplot(ranking[:8],
                        "CLIP-QDA global: {} vs {}".format(c1, c2),
                        os.path.join(xai_dir,
                                     "global_{}_vs_{}.png".format(c1, c2)),
                        xlabel="signed Wasserstein-2")
    save_json(global_out, os.path.join(xai_dir, "global_explanations.json"))

    # Local / LIME / SHAP on a few test samples
    test_df = data["test_df"]
    X_test = data["X_test"]
    rng = np.random.RandomState(cfg.get("seed", 42))
    n = min(args.num_samples, len(X_test))
    idxs = rng.choice(len(X_test), n, replace=False) if len(X_test) else []

    per_sample = []
    for rank, i in enumerate(idxs):
        z = X_test[i]
        true_label = test_df.iloc[i]["label"]
        pred = clf.predict(z[None, :])[0]
        image_id = test_df.iloc[i]["image_id"]
        print("[03] sample {} ({}) true={} pred={}".format(
            rank, image_id, true_label, pred))

        local = local_counterfactuals(z, clf, concepts)[:5]
        lime_list, _ = clip_lime(z, clf, data["X_train"], concepts,
                                 class_names, num_samples=args.lime_samples,
                                 seed=cfg.get("seed", 42))
        shap_list, _ = clip_shap(z, clf, data["X_train"], concepts,
                                 nsamples=args.shap_nsamples,
                                 seed=cfg.get("seed", 42))

        lime_top = sorted(lime_list, key=lambda kv: -abs(kv[1]))[:5]
        shap_top = sorted(shap_list, key=lambda kv: -abs(kv[1]))[:5]

        prefix = os.path.join(xai_dir, "sample{}_{}".format(rank, image_id))
        if local:
            barplot([(d["concept"] + " (->" + d["target_class"] + ")",
                      d["epsilon_scaled"]) for d in local],
                    "CLIP-QDA local (pred={})".format(pred),
                    prefix + "_local.png", xlabel="scaled counterfactual")
        barplot(lime_top, "CLIP-LIME (pred={})".format(pred),
                prefix + "_lime.png", xlabel="LIME weight")
        barplot(shap_top, "CLIP-SHAP (pred={})".format(pred),
                prefix + "_shap.png", xlabel="SHAP value")

        per_sample.append({
            "image_id": str(image_id),
            "true_label": str(true_label),
            "pred_label": str(pred),
            "clip_qda_local": local,
            "clip_lime_top5": [[k, float(v)] for k, v in lime_top],
            "clip_shap_top5": [[k, float(v)] for k, v in shap_top],
        })

    save_json({"samples": per_sample},
              os.path.join(xai_dir, "sample_explanations.json"))
    print("[03] XAI artifacts written to", xai_dir)


if __name__ == "__main__":
    main()
