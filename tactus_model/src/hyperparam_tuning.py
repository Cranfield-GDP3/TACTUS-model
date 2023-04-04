import numpy as np
from tactus_data import skeletonization, data_augment


def get_data_augment_grid():
    pass


base_search_grid = {
    "data_augment": get_data_augment_grid(),
    "window_size": [1, 3, 5, 8, 12],  # impact velocity 
    "number_of_angles": [0, 4, 8],
}

classifiers_hyperparams = {
    "lstm": {
        "..."
    },
    "fc": {
        "..."
    },
}

for classifier, hyperparams in classifiers_hyperparams.items():
    full_grid = hyperparams | base_search_grid

    if classifier == "lstm":
        del full_grid["window_size"]
        del full_grid["velocity"]
