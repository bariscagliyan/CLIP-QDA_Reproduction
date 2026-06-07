"""Dataset loading for MonuMAI and Cats/Dogs/Cars.

Both loaders return a pandas DataFrame describing the samples plus a callable
``get_image(row) -> PIL.Image`` so that CLIP scoring is decoupled from how the
images are stored (local files for MonuMAI, in-memory HF dataset for CDC).

DataFrame schema (consistent across datasets):
    image_id   : str    unique id
    image_path : str    path on disk ("" when loaded from a HF dataset)
    hf_index   : int    index into the HF dataset (-1 for file-based datasets)
    label      : str    class name used by the classifier
    split      : str    "train" or "test"
"""

import glob
import os
import subprocess

import pandas as pd
from PIL import Image

from utils import stratified_split

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")

# Keyword -> canonical MonuMAI class. Folder names in the wild vary in spelling
# and separators, so we match by keyword.
MONUMAI_CLASS_KEYWORDS = [
    ("Hispanic muslim", ("hispanic", "muslim", "mudejar", "arab", "nazari")),
    ("Renaissance", ("renaissance", "rennaissance", "renacimiento")),
    ("Baroque", ("baroque", "barroco", "barroque")),
    ("Gothic", ("gothic", "gotico", "gothique")),
]


def _canonical_monumai_class(name):
    """Map an arbitrary folder/label string to a MonuMAI class, or None."""
    low = name.lower()
    for canonical, keys in MONUMAI_CLASS_KEYWORDS:
        if any(k in low for k in keys):
            return canonical
    return None


# MonuMAI
def try_clone_monumai(dest):
    """Attempt to clone the OD-MonuMAI repository into ``dest``.

    Returns the path that should be scanned for images, or None on failure.
    """
    if os.path.isdir(dest) and glob.glob(os.path.join(dest, "**", "*.jpg"),
                                          recursive=True):
        return dest
    url = "https://github.com/ari-dasci/OD-MonuMAI.git"
    print("[data] Cloning {} -> {}".format(url, dest))
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, dest],
                       check=True)
        return dest
    except Exception as exc:  # noqa: BLE001
        print("[data] Automatic clone failed:", exc)
        return None


def _find_split_file(data_root):
    """Look for an original train/test split shipped with the dataset.

    Returns a dict ``{basename_lower: 'train'|'test'}`` or None.
    Supports a CSV with columns (image/filename/path, split) or pairs of
    train*/test* text files listing image names.
    """
    # 1) CSV with a split column.
    for csv_path in glob.glob(os.path.join(data_root, "**", "*.csv"),
                              recursive=True):
        try:
            df = pd.read_csv(csv_path)
        except Exception:  # noqa: BLE001
            continue
        cols = {c.lower(): c for c in df.columns}
        name_col = next((cols[c] for c in
                         ("image", "filename", "file", "path", "name")
                         if c in cols), None)
        split_col = next((cols[c] for c in ("split", "set", "partition")
                          if c in cols), None)
        if name_col and split_col:
            mapping = {}
            for _, r in df.iterrows():
                base = os.path.basename(str(r[name_col])).lower()
                mapping[base] = ("test" if "test" in str(r[split_col]).lower()
                                 else "train")
            if mapping:
                print("[data] Using original split from", csv_path)
                return mapping
    return None


def load_monumai(data_root=None, test_size=0.2, seed=42, clone_dir=None):
    """Load MonuMAI as a DataFrame + image getter.

    If ``data_root`` is None we try to clone OD-MonuMAI into ``clone_dir``.
    Class is inferred from the parent folder name of each image.
    """
    if data_root is None:
        clone_dir = clone_dir or os.path.join("data", "OD-MonuMAI")
        data_root = try_clone_monumai(clone_dir)
        if data_root is None:
            raise RuntimeError(
                "Could not obtain MonuMAI automatically. Download it manually "
                "and pass --data-root pointing to a folder whose images live "
                "in per-style subfolders (Baroque/Gothic/Hispanic*/Renaissance)."
            )

    rows = []
    for path in glob.glob(os.path.join(data_root, "**", "*"), recursive=True):
        if not path.lower().endswith(IMAGE_EXTS):
            continue
        # Use the nearest folder name(s) on the path to determine the class.
        parts = os.path.normpath(path).split(os.sep)
        label = None
        for part in reversed(parts[:-1]):
            label = _canonical_monumai_class(part)
            if label is not None:
                break
        if label is None:
            continue
        rows.append({
            "image_id": os.path.splitext(os.path.basename(path))[0],
            "image_path": os.path.abspath(path),
            "hf_index": -1,
            "label": label,
        })

    if not rows:
        raise RuntimeError(
            "No MonuMAI images with a recognised class folder were found under "
            "'{}'. Expected subfolders named after the architectural styles."
            .format(data_root))

    df = pd.DataFrame(rows).drop_duplicates("image_path").reset_index(drop=True)

    split_map = _find_split_file(data_root)
    if split_map:
        df["split"] = df["image_path"].apply(
            lambda p: split_map.get(os.path.basename(p).lower(), "train"))
        df.attrs["split_origin"] = "original"
    else:
        df["split"] = stratified_split(df["label"].values, test_size, seed)
        df.attrs["split_origin"] = "stratified_80_20"

    def get_image(row):
        return Image.open(row["image_path"]).convert("RGB")

    return df, get_image


# Cats / Dogs / Cars

_CDC_HF_REPO = "ENSTA-U2IS/Cats_Dogs_Cars"
_CDC_HF_FILE = "Cats_dogs_cars_HF.zip"
# Keyword map for the inner folder name to a classifier class.
_CDC_CLASS_MAP = {"cars": "Car", "cats": "Cat", "dogs": "Dog"}


def _download_cdc_zip(dest_dir):
    """Download the CDC zip from HF hub and extract into dest_dir. Returns dest_dir."""
    import zipfile
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        repo_id=_CDC_HF_REPO, filename=_CDC_HF_FILE, repo_type="dataset"
    )
    print("[data] Extracting CDC zip -> {}".format(dest_dir))
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    return dest_dir


def _find_cdc_root(dest_dir):
    """Locate the inner dataset folder that contains Black/ and White/ subdirs."""
    for root, dirs, _ in os.walk(dest_dir):
        if "Black" in dirs or "White" in dirs:
            return root
    return None


def load_cats_dogs_cars(hf_dataset="ENSTA-U2IS/Cats_Dogs_Cars",
                        local_dir=None, test_size=0.2, seed=42,
                        max_samples=None):
    """Load Cats/Dogs/Cars, downloading from HF if needed.

    Images live under {color}/{species}/ folders after extraction. We assign:
      label    = coarse class (Cat / Dog / Car)
      raw_label = fine-grained (e.g. 'Black Cats')
    """
    if local_dir is None:
        local_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "Cats_Dogs_Cars",
        )

    # Check if already extracted.
    dataset_root = _find_cdc_root(local_dir)
    if dataset_root is None:
        _download_cdc_zip(local_dir)
        dataset_root = _find_cdc_root(local_dir)
    if dataset_root is None:
        raise RuntimeError(
            "Could not find Cats_Dogs_Cars image folders under {}.".format(local_dir))

    rows = []
    for path in glob.glob(os.path.join(dataset_root, "**", "*"), recursive=True):
        if not path.lower().endswith(IMAGE_EXTS):
            continue
        parts = os.path.normpath(path).split(os.sep)
        if len(parts) < 3:
            continue
        species_folder = parts[-2].lower()   # Cars / Cats / Dogs
        color_folder   = parts[-3]           # Black / White
        label = _CDC_CLASS_MAP.get(species_folder)
        if label is None:
            continue
        raw_label = "{} {}".format(color_folder, parts[-2])
        rows.append({
            "image_id": "cdc_" + os.path.splitext(os.path.basename(path))[0],
            "image_path": os.path.abspath(path),
            "hf_index": -1,
            "label": label,
            "raw_label": raw_label,
        })

    if not rows:
        raise RuntimeError("No CDC images found under {}.".format(dataset_root))

    df = pd.DataFrame(rows).drop_duplicates("image_path").reset_index(drop=True)

    if max_samples is not None and len(df) > max_samples:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(
                min(len(g), max(1, max_samples // df["label"].nunique())),
                random_state=seed,
            )
        ).reset_index(drop=True)

    df["split"] = stratified_split(df["label"].values, test_size, seed)
    df.attrs["split_origin"] = "stratified_80_20"

    def get_image(row):
        return Image.open(row["image_path"]).convert("RGB")

    return df, get_image


def load_dataset_by_name(cfg):
    """Dispatch loader based on ``cfg['dataset']``. Returns (df, get_image)."""
    name = cfg["dataset"].lower()
    if name == "monumai":
        return load_monumai(
            data_root=cfg.get("data_root"),
            test_size=cfg.get("test_size", 0.2),
            seed=cfg.get("seed", 42),
        )
    if name in ("cats_dogs_cars", "cats-dogs-cars", "cdc"):
        return load_cats_dogs_cars(
            local_dir=cfg.get("local_dir"),
            test_size=cfg.get("test_size", 0.2),
            seed=cfg.get("seed", 42),
            max_samples=cfg.get("max_samples"),
        )
    raise ValueError("Unknown dataset: {}".format(cfg["dataset"]))
