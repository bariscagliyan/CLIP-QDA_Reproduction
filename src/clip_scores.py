"""Compute and cache CLIP concept scores.

A CLIP score is ``score_scale * cosine_similarity(image_embedding,
concept_text_embedding)`` (Section 3.1 of the paper). The image and text
embeddings are L2-normalised before the dot product, so the cosine similarity
lies in [-1, 1]; multiplying by ``score_scale`` (default 100, the CLIP logit
scale) reproduces the score magnitudes reported in the paper (~10-25).
"""

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from concepts import get_concept_set
from utils import ensure_dir


def load_clip(model_name="ViT-L/14@336px", device="cpu"):
    """Load an OpenAI CLIP model and its preprocessing transform."""
    import clip  # openai-clip
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


@torch.no_grad()
def encode_concepts(model, concepts, prompt_template="{concept}", device="cpu"):
    """Return L2-normalised text embeddings for every concept (N, D)."""
    import clip
    prompts = [prompt_template.format(concept=c) for c in concepts]
    tokens = clip.tokenize(prompts).to(device)
    feats = model.encode_text(tokens).float()
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def column_names(concepts):
    """Safe, unique CSV column names for the concepts."""
    seen, cols = {}, []
    for c in concepts:
        name = "concept::" + c
        if name in seen:
            seen[name] += 1
            name = "{}__{}".format(name, seen[name])
        else:
            seen[name] = 0
        cols.append(name)
    return cols


@torch.no_grad()
def extract_scores(df, get_image, cfg, device="cpu"):
    """Compute the concept-score table for every row of ``df``.

    Returns a DataFrame with columns:
        image_id, image_path, hf_index, label, split, <one per concept>.
    """
    info = get_concept_set(cfg["dataset"])
    concepts = info["concepts"]
    score_scale = float(cfg.get("score_scale", 100.0))
    batch_size = int(cfg.get("batch_size", 4))

    model, preprocess = load_clip(cfg.get("clip_model", "ViT-L/14@336px"),
                                  device)
    text_feats = encode_concepts(model, concepts,
                                 cfg.get("prompt_template", "{concept}"),
                                 device)

    cols = column_names(concepts)
    records = []
    rows = df.to_dict("records")

    for start in tqdm(range(0, len(rows), batch_size), desc="CLIP scoring"):
        batch = rows[start:start + batch_size]
        images = torch.stack([preprocess(get_image(r)) for r in batch]).to(device)
        img_feats = model.encode_image(images).float()
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        sims = (img_feats @ text_feats.t()) * score_scale  # (B, N)
        sims = sims.cpu().numpy()
        for r, s in zip(batch, sims):
            rec = {
                "image_id": r["image_id"],
                "image_path": r.get("image_path", ""),
                "hf_index": r.get("hf_index", -1),
                "label": r["label"],
                "split": r["split"],
            }
            if "raw_label" in r:
                rec["raw_label"] = r["raw_label"]
            rec.update({col: float(v) for col, v in zip(cols, s)})
            records.append(rec)

    return pd.DataFrame(records)


def save_scores(scores_df, path):
    ensure_dir(path)
    scores_df.to_csv(path, index=False)
    print("[clip_scores] wrote", path, "({} rows)".format(len(scores_df)))


def load_scores(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Concept-score cache not found: {}. Run "
            "scripts/01_extract_concept_scores.py first.".format(path))
    return pd.read_csv(path)


def scores_matrix(scores_df):
    """Split a cached score table into train/test arrays.

    Returns dict with X_train, y_train, X_test, y_test, concept_cols.
    """
    concept_cols = [c for c in scores_df.columns if c.startswith("concept::")]
    train = scores_df[scores_df["split"] == "train"]
    test = scores_df[scores_df["split"] == "test"]
    return {
        "concept_cols": concept_cols,
        "X_train": train[concept_cols].to_numpy(dtype=np.float64),
        "y_train": train["label"].to_numpy(),
        "X_test": test[concept_cols].to_numpy(dtype=np.float64),
        "y_test": test["label"].to_numpy(),
        "train_df": train.reset_index(drop=True),
        "test_df": test.reset_index(drop=True),
    }
