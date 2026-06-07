"""Small shared helpers: config loading, seeding, device, IO, splitting."""

import json
import os
import random

import numpy as np
import yaml


def load_config(path):
    """Load a YAML config file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


def set_seed(seed=42):
    """Set deterministic seeds across random, numpy and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_device():
    """Return 'cuda' if a GPU is available else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_dir(path):
    """Create the directory holding ``path`` (or ``path`` itself) if missing."""
    d = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def save_json(obj, path):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stratified_split(labels, test_size=0.2, seed=42):
    """Deterministic stratified train/test split.

    Returns an array of strings ("train"/"test") aligned with ``labels``.
    """
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    split = np.array(["train"] * len(labels), dtype=object)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        # guarantee at least one test and one train sample per class
        n_test = min(max(n_test, 1), max(len(idx) - 1, 1))
        split[idx[:n_test]] = "test"
    return split


def project_root():
    """Absolute path to the repository root (parent of ``src``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
