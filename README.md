# CLIP-QDA: Reproduction

CENG502 reproduction project.

Paper: "CLIP-QDA: An Explainable Concept Bottleneck Model", Remi Kazmierczak,
Eloise Berthier, Goran Frehse, Gianni Franchi. Transactions on Machine Learning
Research (TMLR), 05/2024. https://openreview.net/forum?id=jjmdiMiag7


## 1. Paper / method description

CLIP-QDA is a Concept Bottleneck Model (CBM) built on top of the frozen CLIP
model. It classifies images in two transparent stages (Figure 2 of the paper):

1. Concept bottleneck. A fixed set of human-readable textual concepts
   k_1, ..., k_N is defined per dataset. For an image x, the model computes a
   CLIP score for each concept: the cosine similarity between the CLIP image
   embedding of x and the CLIP text embedding of the concept, scaled by a
   constant. This gives a low-dimensional, interpretable concept-score vector
   z = [z_1, ..., z_N].

2. QDA classifier. The paper observes that, for a small number of well-chosen
   concepts, the per-class distribution of z is approximately a multivariate
   Gaussian. CLIP-QDA models each class c as N(z | mu_c, Sigma_c) with prior
   p_c and classifies with the maximum-a-posteriori rule (Eqs. 1-4). Because
   the parameters are simple statistics, training is immediate.

The paper also derives explainability methods from this formulation (Sec. 3.4):

- CLIP-QDA_local: closed-form, sparse, signed counterfactuals
  (Proposition 1 / Appendix A.5), i.e. how much a concept must change to flip
  the predicted label.
- CLIP-LIME / CLIP-SHAP: LIME and Kernel SHAP on the concept-score vector.
- CLIP-QDA_global: signed Wasserstein-2 distance between class Gaussians
  (used here for qualitative figures).

Explanation quality is measured with two metrics (Sec. 4.3): Deletion (Del,
Eq. 7) for faithfulness to the model, and Detection (Det, Eq. 8) for
faithfulness to the data.


## 2. What we reproduced


| # | Target | Scope |
|---|--------|-------|
| 1 | Core pipeline: image -> CLIP concept scores -> concept-score table -> QDA 
| 2 | Table 2, one cell: CLIP-QDA accuracy on MonuMAI (paper = 0.89) 
| 3 | The same pipeline applied to Cats/Dogs/Cars 
| 4 | Intermediate concept-score tables (one row per image, one column per concept) 
| 5 | Table 4 (partial): XAI Detection metric for CLIP-QDA_local, CLIP-LIME, CLIP-SHAP 

Not part of this work: ImageNet, PASCAL-Part, MIT scenes, Flowers-102; the
LaBo, Yan et al., ResNet and ViT baselines; and the LaBo/Yan/Random rows of
Table 4.

Settings: CLIP ViT-L/14@336px (the paper's model), CLIP score = 100 x cosine
similarity, prompt = the raw concept string, seed 42, batch_size 32
(inference only, fits the RTX 3060 6 GB), QDA reg_param swept over
{0, 1e-4, 1e-3, 1e-2}.

Train/test split: the paper uses each dataset's original split, which is not
separately published for MonuMAI. We use a deterministic stratified 80/20 split
(seed 42) for both datasets, recorded in the split_origin field of the outputs.
Reported accuracy may differ from 0.89 because of split differences,
prompt/scoring choices, preprocessing, and QDA regularization.


## 3. Results obtained

### 3.1 Main target: Table 2, MonuMAI (paper CLIP-QDA = 0.89)

Ours:

| Dataset | Method | Paper acc. | Reproduced acc. | reg_param | Delta vs paper | Note |
|---------|--------|-----------|-----------------|-----------|---------------|------|
| MonuMAI | CLIP-QDA (default reg) | 0.89 | 0.8510 | 1e-4 | -0.0390 | stratified 80/20 |
| MonuMAI | CLIP-QDA (best of sweep) | 0.89 | 0.8543 | 1e-2 | -0.0357 | stratified 80/20 |
| Cats/Dogs/Cars | CLIP-QDA | - | 0.9984 | any | - | secondary |

Paper (Table 2, relevant cells):

| Method | PASCAL-Part | MIT scenes | MonuMAI | ImageNet |
|--------|-------------|------------|---------|----------|
| CLIP-QDA | 0.90 | 0.81 | 0.89 | 0.60 |

Our MonuMAI accuracy (0.85) is within about 4 points of the paper's 0.89, which
is consistent with using a different (stratified) train/test split.

### 3.2 Partial target: Table 4, XAI metrics on Cats/Dogs/Cars

We reproduce the Detection (Det) metric, which is the meaningful bias-detection
result. Higher is better. Det is computed as in the paper: averaged over the six
biased binary tasks (Black X + White Y, with different species) using
ground-truth concepts Ss = {Black, White}, and checking whether the bias
concepts appear in the explanation's top-2.

| Method | Det (ours) | Det (paper) |
|--------|-----------|-------------|
| CLIP-QDA_local | 0.0935 | 0.2724 |
| CLIP-LIME | 0.3759 | 0.4042 |
| CLIP-SHAP | 0.3705 | 0.3696 |

CLIP-LIME and CLIP-SHAP match the paper closely (SHAP 0.3705 vs 0.3696, LIME
0.3759 vs 0.4042). Both rank concepts by attribution magnitude, so the
highly-discriminative bias concepts Black/White land in the top-2.
CLIP-QDA_local scores lower (0.09 vs 0.27): it ranks concepts by counterfactual
proximity to the decision boundary, and the bias concepts are far from the
boundary (the classes are well separated along them), so they rank as less
important. Local being the weakest of the three at bias detection is consistent
with the paper (paper local 0.27 is also below SHAP/LIME).

Deletion (Del) is reported only as an internal sanity check, not a paper
reproduction. The output file gives del_method_order vs del_random_order, which
confirms that deleting concepts in the method's importance order degrades
accuracy faster than a random order. These do not correspond to the paper's
Del Set 1 / Set 2, which are two concept vocabularies (the real Table-6 concepts
vs an equal number of random dictionary words) evaluated with each method's
ordering.


## 4. Code structure

```
clip-qda-reproduction/
  README.md
  requirements.txt
  run_reproduce.sh             end-to-end driver
  configs/
    monumai.yaml               MonuMAI config (backbone, reg sweep, paths)
    cats_dogs_cars.yaml        Cats/Dogs/Cars config
  src/
    concepts.py                concept sets from Table 6 of the paper
    data.py                    MonuMAI (git clone/scan) + Cats/Dogs/Cars loaders
    clip_scores.py             CLIP scoring (scaled cosine) + CSV cache
    qda_model.py               CLIP-QDA classifier (sklearn QDA wrapper)
    xai.py                     local counterfactuals, CLIP-LIME, CLIP-SHAP,
                               global Wasserstein, Del/Det metrics
    utils.py                   config, seeding, device, IO, stratified split
  scripts/
    00_prepare_data.py         download / verify both datasets
    01_extract_concept_scores.py   CLIP to concept-score CSV (cached)
    02_train_eval_qda.py       train + evaluate QDA (Table 2 / accuracy)
    03_run_xai_cats_dogs_cars.py   qualitative XAI figures
    04_make_results_tables.py  print summary tables
    05_table4_xai_metrics.py   partial Table 4 (Det, plus Del sanity check)
  results/                     all CSV / JSON / figure outputs
```

The expensive CLIP forward pass (step 01) runs once and is cached to CSV. The
QDA classifier (02), qualitative XAI (03) and Table 4 metrics (05) all re-read
those cached concept-score tables, so they can be re-run without recomputing
CLIP.


## 5. Installation and running

### 5.1 Install (Python 3.10 recommended)

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux / macOS:         source .venv/bin/activate

pip install -r requirements.txt
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # expect: True
```

The code uses CUDA when available and falls back to CPU.

### 5.2 Run everything

Git Bash / WSL / Linux / macOS:
```bash
bash run_reproduce.sh
```

### 5.3 Run step by step

Windows PowerShell:
```powershell
python scripts\00_prepare_data.py
python scripts\01_extract_concept_scores.py --config configs\monumai.yaml
python scripts\01_extract_concept_scores.py --config configs\cats_dogs_cars.yaml
python scripts\02_train_eval_qda.py --config configs\monumai.yaml
python scripts\02_train_eval_qda.py --config configs\cats_dogs_cars.yaml
python scripts\03_run_xai_cats_dogs_cars.py
python scripts\04_make_results_tables.py
python scripts\05_table4_xai_metrics.py --shap-nsamples 50
```

Git Bash / Linux / macOS:
```bash
python scripts/00_prepare_data.py
python scripts/01_extract_concept_scores.py --config configs/monumai.yaml
python scripts/01_extract_concept_scores.py --config configs/cats_dogs_cars.yaml
python scripts/02_train_eval_qda.py --config configs/monumai.yaml
python scripts/02_train_eval_qda.py --config configs/cats_dogs_cars.yaml
python scripts/03_run_xai_cats_dogs_cars.py
python scripts/04_make_results_tables.py
python scripts/05_table4_xai_metrics.py --shap-nsamples 50
```

```bash
python scripts/01_extract_concept_scores.py --config configs/monumai.yaml --data-root /path/to/MonuMAI
```

### 5.4 Datasets

- MonuMAI: auto-cloned from ari-dasci/OD-MonuMAI; class is inferred from each
  image's parent folder.
- Cats/Dogs/Cars: from Hugging Face ENSTA-U2IS/Cats_Dogs_Cars (auto-downloaded).
  The six fine-grained categories (Black/White by Cat/Dog/Car) are mapped to
  three classifier classes (Cat, Dog, Car); the bias concepts Black and White
  are added for the XAI experiments.

### 5.5 Output files

| File | Contents |
|------|----------|
| results/concept_scores/monumai_scores.csv | cached MonuMAI concept scores (1514 rows, 20 concepts) |
| results/concept_scores/cats_dogs_cars_scores.csv | cached CDC concept scores (6436 rows, 17 concepts) |
| results/table2_monumai_reproduction.csv / .json | MonuMAI accuracy vs paper, per reg_param |
| results/cats_dogs_cars_accuracy.csv / .json | Cats/Dogs/Cars accuracy sweep |
| results/table4_xai_metrics.csv / .json | partial Table 4: Det (vs paper) and Del sanity check |
| results/xai_cats_dogs_cars/ | qualitative XAI figures + JSON |

Concept-score CSVs have columns:
image_id, image_path, hf_index, label, split, split_origin, concept::<name>...

### 5.6 Concepts used (Table 6 of the paper)

MonuMAI, 4 classes x 5 = 20 concepts:

| Class | Concepts |
|-------|----------|
| Baroque | Ornate, Elaborate sculptures, Intricate details, Curved or asymmetrical design, Historical or aged appearance |
| Gothic | Pointed arches, Ribbed vaults, Flying buttresses, Stained glass windows, Tall spires or towers |
| Hispanic muslim | Mudejar style, Intricate geometric patterns, Horseshoe arches, Decorative tilework azulejos, Islamic-inspired motifs |
| Renaissance | Classical proportions, Symmetrical design, Columns and pilasters, Human statues and sculptures, Dome or dome-like structures |

Cats/Dogs/Cars, 3 classes x 5 + 2 bias = 17 concepts:

| Class | Concepts |
|-------|----------|
| Cat | Furry, Whiskered, Pointy-eared, Slitted-eyed, Four-legged |
| Car | Metallic, Four-wheeled, Headlights, Windshield, License plate |
| Dog | Snout, Wagging-tailed, Snout-nosed, Floppy-eared, Tail-wagging |
| bias | Black, White |
