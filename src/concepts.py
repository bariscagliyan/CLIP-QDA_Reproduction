"""Concept sets used by CLIP-QDA.

The concepts are taken verbatim from Table 6 of the paper.

Each dataset defines a mapping ``class name -> list of 5 visual descriptors``.
The CLIP score of an image against every descriptor forms the concept-score
vector that is fed to the QDA classifier.
"""

from collections import OrderedDict

# MonuMAI: 4 architectural styles x 5 concepts = 20 concepts
MONUMAI_CONCEPTS = OrderedDict([
    ("Baroque", [
        "Ornate",
        "Elaborate sculptures",
        "Intricate details",
        "Curved or asymmetrical design",
        "Historical or aged appearance",
    ]),
    ("Gothic", [
        "Pointed arches",
        "Ribbed vaults",
        "Flying buttresses",
        "Stained glass windows",
        "Tall spires or towers",
    ]),
    ("Hispanic muslim", [
        "Mudejar style",
        "Intricate geometric patterns",
        "Horseshoe arches",
        "Decorative tilework azulejos",
        "Islamic-inspired motifs",
    ]),
    ("Renaissance", [
        "Classical proportions",
        "Symmetrical design",
        "Columns and pilasters",
        "Human statues and sculptures",
        "Dome or dome-like structures",
    ]),
])


# Cats/Dogs/Cars: 3 classes x 5 concepts = 15 concepts.
# Two extra "bias" concepts (Black, White) are appended. 

CATS_DOGS_CARS_CONCEPTS = OrderedDict([
    ("Cat", [
        "Furry",
        "Whiskered",
        "Pointy-eared",
        "Slitted-eyed",
        "Four-legged",
    ]),
    ("Car", [
        "Metallic",
        "Four-wheeled",
        "Headlights",
        "Windshield",
        "License plate",
    ]),
    ("Dog", [
        "Snout",
        "Wagging-tailed",
        "Snout-nosed",
        "Floppy-eared",
        "Tail-wagging",
    ]),
])

# Bias concepts (added on top of the class concepts).
BIAS_CONCEPTS = ["Black", "White"]


def get_concept_set(dataset):
    """Return concept metadata for a dataset.

    Parameters
    dataset : str
        Either ``"monumai"`` or ``"cats_dogs_cars"``.

    Returns
    dict with keys:
        class_names    : list[str]   ordered class labels for the classifier
        concepts       : list[str]   flat ordered list of concept strings
        concept_owner  : list[str]   owning class for each concept ("" for bias)
        is_bias        : list[bool]  whether each concept is a bias concept
    """
    dataset = dataset.lower()
    if dataset == "monumai":
        mapping = MONUMAI_CONCEPTS
        bias = []
    elif dataset in ("cats_dogs_cars", "cats-dogs-cars", "cdc"):
        mapping = CATS_DOGS_CARS_CONCEPTS
        bias = BIAS_CONCEPTS
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    class_names = list(mapping.keys())
    concepts, owner, is_bias = [], [], []
    for cls, cs in mapping.items():
        for c in cs:
            concepts.append(c)
            owner.append(cls)
            is_bias.append(False)
    for c in bias:
        concepts.append(c)
        owner.append("")
        is_bias.append(True)

    return {
        "class_names": class_names,
        "concepts": concepts,
        "concept_owner": owner,
        "is_bias": is_bias,
    }


if __name__ == "__main__":
    for ds in ("monumai", "cats_dogs_cars"):
        info = get_concept_set(ds)
        print(ds, "->", len(info["concepts"]), "concepts,",
              len(info["class_names"]), "classes")
        print("  classes:", info["class_names"])
        print("  concepts:", info["concepts"])
