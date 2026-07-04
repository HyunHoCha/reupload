from __future__ import annotations

LEARNING_RATE = 0.01
COMPLEX_DTYPE_NAME = "complex64"

POLY_NUM_EPOCHS = 10_000
POLY_GRID_POINTS = 201
POLY_GRID_MIN = -1.0
POLY_GRID_MAX = 1.0
POLY_TARGETS = {
    "lambda": {
        "L": 1,
        "title": r"$\lambda$",
        "coeffs_desc": [1.0, 0.0],
    },
    "quartic": {
        "L": 4,
        "title": r"$3(\lambda+0.8)\lambda(\lambda-0.5)^2+0.3$",
        "coeffs_desc": [3.0, -0.6, -1.65, 0.6, 0.3],
    },
    "cos": {
        "L": 2,
        "title": r"$\cos\lambda$",
        "coeffs_desc": None,
    },
}
CONSTRUCTIVE_DELTAS = {
    "lambda": [1.0, 1 / 2, 1 / 3, 1 / 4],
    "quartic": [1 / 3, 1 / 5, 1 / 10, 1 / 100],
}

CLASSIFICATION_NUM_EPOCHS = 1_000
TRAIN_SIZE = 1_000
TEST_SIZE = 500
PURITY_THRESHOLD = (1.0 + 2.0 ** (-2.0 / 3.0)) / 2.0
ENTANGLEMENT_ENTROPY_THRESHOLD = 0.3
CLASSIFIER_DEPTHS = [1, 2, 3, 4]
BLOCH_RULE_A_DEPTHS = [1, 2]
BLOCH_RULE_B_DEPTHS = [2, 3]

CLASSIFICATION_OUTPUT_THRESHOLD = 0.5

DEFAULT_SEED = 0
