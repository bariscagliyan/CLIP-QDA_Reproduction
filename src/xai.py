"""Explainability methods for CLIP-QDA.

Implements the four explanation families from the paper:

  * CLIP-QDA_global : signed Wasserstein-2 distance between the per-class
                      Gaussian marginals of a concept (Section 3.4.1).
  * CLIP-QDA_local  : closed-form sparse, signed counterfactuals from
                      Proposition 1 / Appendix A.5 (Sections 3.4.2).
  * CLIP-LIME       : LIME on the tabular concept-score vector (Section 3.4.3).
  * CLIP-SHAP       : Kernel SHAP on the tabular concept-score vector.

Also implements the quantitative evaluation metrics from Section 4.3:

  * deletion_metric  : Del score (Eq. 7), faithfulness to the model.
  * detection_metric : Det score (Eq. 8), faithfulness to the data.
"""

import numpy as np


# CLIP-QDA global: signed Wasserstein-2 between class marginals of a concept.
def signed_wasserstein2(clf, c1, c2):
    """Signed W2 distance per concept between classes c1 and c2 (Eq. of 3.4.1).
    Positive value => concept mean is higher for c1 than c2.
    """
    i1, i2 = clf.class_index(c1), clf.class_index(c2)
    mu1, mu2 = clf.means_[i1], clf.means_[i2]
    var1 = np.diag(clf.covariances_[i1])
    var2 = np.diag(clf.covariances_[i2])
    dmu = mu1 - mu2
    lam = var1 + var2 - 2.0 * np.sqrt(np.clip(var1 * var2, 0, None))
    val = np.sqrt(np.clip(dmu ** 2 + lam, 0, None))
    return np.sign(dmu) * val


def global_explanation(clf, c1, c2, concepts, top_k=5):
    """Rank concepts by |signed W2| for the (c1, c2) class pair."""
    sw = signed_wasserstein2(clf, c1, c2)
    order = np.argsort(-np.abs(sw))
    return [(concepts[i], float(sw[i])) for i in order[:top_k]]


# CLIP-QDA local: closed-form counterfactuals (Proposition 1 / Appendix A.5).
def _binary_counterfactual(z, j, sign, clf, ch, ci):
    """Solve problem (5) for the binary case ch vs ci, concept j, given sign.

    Returns the perturbation epsilon (float) or None. Sign is +1 or -1.
    Closed form from Eqs. (10)-(11) and (17) in Appendix A.5.
    """
    mu_h, mu_i = clf.means_[ch], clf.means_[ci]
    S_h = clf.covariances_[ch]
    S_i = clf.covariances_[ci]
    Si_h = np.linalg.inv(S_h)
    Si_i = np.linalg.inv(S_i)
    p = clf.priors_

    P = 0.5 * (Si_i[j, j] - Si_h[j, j])
    b = float(np.sum((z - mu_i) * Si_i[j, :] - (z - mu_h) * Si_h[j, :]))
    sign_det = np.linalg.slogdet(S_h)[1] - np.linalg.slogdet(S_i)[1]
    c = (0.5 * sign_det
         + np.log(p[ch]) - np.log(p[ci])
         + 0.5 * (z - mu_i) @ Si_i @ (z - mu_i)
         - 0.5 * (z - mu_h) @ Si_h @ (z - mu_h))

    candidates = []
    if abs(P) < 1e-12:
        # Linear case: b*eps + c = 0.
        if abs(b) > 1e-12:
            candidates.append(-c / b)
    else:
        disc = b * b - 4.0 * P * c
        if disc > 0:
            sq = np.sqrt(disc)
            candidates.append((-b - sq) / (2.0 * P))
            candidates.append((-b + sq) / (2.0 * P))

    # Keep candidates matching the requested sign; pick minimal magnitude.
    valid = [e for e in candidates if np.sign(e) == sign and abs(e) > 1e-9]
    if not valid:
        return None
    return min(valid, key=abs)


def local_counterfactuals(z, clf, concepts, scale=True):
    """All sparse signed counterfactuals for a sample (Proposition 1, Sec 3.4.2).

    For each concept j and sign s, returns the minimal-magnitude perturbation
    that flips the predicted label (multiclass extended per Appendix A.5: solve
    C-1 binary sub-problems and take the smallest-magnitude result).

    Returns a list of dicts sorted by |scaled magnitude| ascending.
    """
    z = np.asarray(z, dtype=np.float64)
    pred = clf.predict(z[None, :])[0]
    ch = clf.class_index(pred)
    other = [k for k in range(len(clf.classes_)) if k != ch]
    var_h = np.diag(clf.covariances_[ch])

    out = []
    for j in range(len(concepts)):
        for sign in (+1, -1):
            best = None
            for ci in other:
                eps = _binary_counterfactual(z, j, sign, clf, ch, ci)
                if eps is None:
                    continue
                # verify the flip actually happens for the full classifier
                z2 = z.copy()
                z2[j] += eps
                if clf.predict(z2[None, :])[0] == pred:
                    continue
                if best is None or abs(eps) < abs(best[0]):
                    best = (eps, clf.classes_[ci])
            if best is None:
                continue
            eps, target = best
            scaled = eps / np.sqrt(var_h[j]) if scale and var_h[j] > 0 else eps
            out.append({
                "concept": concepts[j],
                "sign": "+" if sign > 0 else "-",
                "epsilon": float(eps),
                "epsilon_scaled": float(scaled),
                "target_class": target,
            })
    out.sort(key=lambda d: abs(d["epsilon_scaled"]))
    return out


# CLIP-LIME and CLIP-SHAP (tabular, operating on the concept scores).
def clip_lime(z, clf, X_train, concepts, class_names, num_samples=1000,
              seed=42):
    """LIME explanation of the QDA decision for sample z (tabular)."""
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_train),
        feature_names=list(concepts),
        class_names=list(class_names),
        discretize_continuous=True,
        kernel_width=0.75 * len(concepts),  # paper: 0.75 * N
        random_state=seed,
        mode="classification",
    )
    pred = clf.predict(np.asarray(z)[None, :])[0]
    label_idx = clf.class_index(pred)
    exp = explainer.explain_instance(
        np.asarray(z, dtype=np.float64),
        clf.predict_proba,
        num_features=len(concepts),
        num_samples=num_samples,
        labels=(label_idx,),
    )
    return exp.as_list(label=label_idx), pred


def clip_shap(z, clf, X_background, concepts, nsamples=200, seed=42):
    """Kernel SHAP explanation of the QDA decision for sample z (tabular)."""
    import shap

    rng = np.random.RandomState(seed)
    bg = np.asarray(X_background, dtype=np.float64)
    if len(bg) > 100:
        bg = bg[rng.choice(len(bg), 100, replace=False)]
    explainer = shap.KernelExplainer(clf.predict_proba, bg)
    z = np.asarray(z, dtype=np.float64)[None, :]
    shap_values = explainer.shap_values(z, nsamples=nsamples, silent=True)

    pred = clf.predict(z)[0]
    label_idx = clf.class_index(pred)
    # shap may return a list (per class) or a 3D array (n, features, classes).
    if isinstance(shap_values, list):
        vals = np.asarray(shap_values[label_idx])[0]
    else:
        arr = np.asarray(shap_values)
        vals = arr[0, :, label_idx] if arr.ndim == 3 else arr[0]
    return list(zip(concepts, [float(v) for v in vals])), pred


# Quantitative evaluation: Deletion and Detection metrics (Section 4.3).

def _null_value(X_train, concept_idx):
    """Average concept score across all training samples (the nullification value)."""
    return float(np.mean(X_train[:, concept_idx]))


def deletion_metric(X_test, y_test, clf, importance_fn, X_train,
                    n_max=9, random_seed=None):
    """Deletion metric (Eq. 7).

    For each test sample the concepts are zeroed-out one-by-one in the order
    returned by ``importance_fn(z, clf, X_train) -> list[int]`` (concept indices,
    most-important first).  Nullification sets a concept to its mean over the
    training set.  Lower Del is better (fewer concept deletions needed to hurt
    accuracy).
    """
    rng = np.random.RandomState(random_seed) if random_seed is not None else None
    n_concepts = X_test.shape[1]
    null_vals = np.array([_null_value(X_train, j) for j in range(n_concepts)])

    # Pre-compute importance orderings once per sample (reused across n_null levels).
    orders = []
    for z in X_test:
        if rng is not None:
            orders.append(rng.permutation(n_concepts).tolist())
        else:
            orders.append(importance_fn(z, clf, X_train))

    accs = np.zeros(n_max + 1)
    for n_null in range(n_max + 1):
        preds = []
        for z, y, order in zip(X_test, y_test, orders):
            z_mod = z.copy()
            for idx in order[:n_null]:
                z_mod[idx] = null_vals[idx]
            preds.append(clf.predict(z_mod[None, :])[0])
        accs[n_null] = np.mean(np.array(preds) == y_test)

    acc0 = accs[0] if accs[0] > 0 else 1.0
    # Normalise by Acc(0) and n_max so Del lies in [0, 1] (paper-scale values).
    del_score = (1.0 / (acc0 * n_max)) * np.sum(
        (accs[:-1] + accs[1:]) / 2.0
    )
    return float(del_score), accs.tolist()


def detection_metric(X_test, clf, importance_fn, X_train, ground_truth_fn,
                     top_k=2):
    """Detection metric (Eq. 8).

    For each test sample, compare the top ``top_k`` concepts from the
    explanation (set Ts) to the ground-truth concept set Ss provided by
    ``ground_truth_fn(predicted_label) -> set[int]``.

    Det = mean over samples of |Ss intersect Ts| / |Ss|.
    """
    scores = []
    for z in X_test:
        pred = clf.predict(z[None, :])[0]
        ss = ground_truth_fn(pred)
        if not ss:
            continue
        order = importance_fn(z, clf, X_train)
        ts = set(order[:top_k])
        scores.append(len(ss & ts) / len(ss))
    return float(np.mean(scores)) if scores else float("nan")


# Importance-function factories used by the metrics 

def importance_local(z, clf, X_train, concepts):
    """Concept indices sorted by |epsilon_scaled| ascending (easiest flip first)."""
    cfs = local_counterfactuals(z, clf, concepts)
    # Build per-concept score: min |epsilon_scaled| over all signs (lower = easier flip).
    best = {}
    for d in cfs:
        idx = concepts.index(d["concept"])
        v = abs(d["epsilon_scaled"])
        if idx not in best or v < best[idx]:
            best[idx] = v
    # Concepts with a counterfactual are more important (lower epsilon).
    n = len(concepts)
    scored = [(best.get(j, np.inf), j) for j in range(n)]
    scored.sort()
    return [j for _, j in scored]


def importance_lime(z, clf, X_train, concepts, class_names, num_samples=500, seed=42):
    """Concept indices sorted by |LIME weight| descending."""
    pairs, _ = clip_lime(z, clf, X_train, concepts, class_names,
                         num_samples=num_samples, seed=seed)
    name_to_idx = {c: i for i, c in enumerate(concepts)}
    # LIME returns "(feature<=threshold, weight)" strings; extract concept name.
    scored = []
    for feat_str, w in pairs:
        for name, idx in name_to_idx.items():
            if name in feat_str:
                scored.append((abs(w), idx))
                break
    scored.sort(reverse=True)
    seen, order = set(), []
    for _, idx in scored:
        if idx not in seen:
            seen.add(idx)
            order.append(idx)
    # Append any remaining concepts not mentioned by LIME.
    order += [j for j in range(len(concepts)) if j not in seen]
    return order


def importance_shap(z, clf, X_train, concepts, nsamples=100, seed=42):
    """Concept indices sorted by |SHAP value| descending."""
    pairs, _ = clip_shap(z, clf, X_train, concepts, nsamples=nsamples, seed=seed)
    scored = sorted(enumerate(pairs), key=lambda kv: -abs(kv[1][1]))
    return [idx for idx, _ in scored]
