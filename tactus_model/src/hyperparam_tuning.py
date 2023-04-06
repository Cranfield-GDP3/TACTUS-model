import json
import numpy as np
from pathlib import Path
from tactus_data import skeletonization, data_augment
from tactus_yolov7 import Yolov7, resize
from utils.tracker import FeatureTracker


DATA_AUGMENT_GRIDS = {
    "BIG_GRID": {
        "noise_amplitude": [0, 3],
        "horizontal_flip": [True, False],
        "rotation_y": [-30, -20, -10, 0, 10, 20, 30],
        "rotation_z": [-10, 0, 10],
        "rotation_x": [-30, -20, -10, 0, 10, 20, 30],
        "scale_x": [0.9, 1.1],
        "scale_y": [0.9, 1.1],
    },
    "DEFAULT_GRID": {
        "noise_amplitude": [0, 3],
        "horizontal_flip": [True, False],
        "rotation_y": [-30, 0, 30],
        "rotation_z": [-10, 0, 10],
        "rotation_x": [-30, 0, 30],
        "scale_x": [0.8, 1, 1.2],
        "scale_y": [0.8, 1, 1.2],
    },
    "MED_GRID": {
        "noise_amplitude": [0, 3],
        "horizontal_flip": [True, False],
        "rotation_y": [-30, -20, -10, 0, 10, 20, 30],
        "rotation_z": [-10, 0, 10],
        "rotation_x": [-30, -20, -10, 0, 10, 20, 30],
    },
    "SMALL_GRID": {
        "noise_amplitude": [0, 2, 4],
        "horizontal_flip": [True, False],
        "rotation_y": [-30, 0, 30],
        "rotation_z": [-10, 0, 10],
        "rotation_x": [-30, 0, 30],
    },
    "SMALLER_GRID": [  # grid size: 26
        {
            "horizontal_flip": [True, False],
            "scale_x": np.linspace(0.8, 1.2, 2),
            "scale_y": np.linspace(0.8, 1.2, 2),
        },
        {
            "horizontal_flip": [True, False],
            "rotation_y": np.linspace(-30, 30, 3),
            "rotation_x": np.linspace(-20, 20, 3),
        },
    ]
}


FEATURES_GRID = {  # grid size: 12
    "window_size": [3, 5, 9, 15],  # impact velocity
    "number_of_angles": [0, 4, 8],
}


CLASSIFIER_HYPERPARAMS = {
    "lstm": {  # grid size: 72
        "batch_size": [64, 256],
        "epochs": [100],
        "neurons": [50, 100, 250, 500],
        "activation": ["tanh", "relu", "sigmoid"],
        "features_size": ["computed later"],
        "dropout_layer": [0, 0.2, 0.4],
    },
    "knn": {
        "n_neighbors": [2, 5, 8, 15, 30, 50],
        "algorithm": ["auto"],
        "metric": ['euclidean', 'manhattan', 'minkowski'],
        "p": [1, 2],
    },
}


def get_grid():
    for classifier, hyperparams in CLASSIFIER_HYPERPARAMS.items():
        for _, augment_grid in DATA_AUGMENT_GRIDS.items():
            yield classifier, hyperparams, augment_grid, FEATURES_GRID


def train(fps: int = 10):
    videos = Path("data/processed/ut_interaction/").iterdir()
    for video_path in videos:
        label_path = video_path / (f"{video_path.stem}.label.json")
        label = json.load(label_path.open())

        offender = label["offender"][0]
        original_data = video_path / f"{fps}fps" / "yolov7.json"

        for classifier, hyperparams, augment_grid, features_grid in get_grid():
            angle_list = get_angle_list(features_grid["number_of_angles"])
            window_size = features_grid["window_size"]
            feature_tracker = FeatureTracker(window_size, angles_to_compute=angle_list)

            for frame in video["frame"]:
                skeletons = yolov7.predict_frame(frame)
                feature_tracker.track_skeletons(skeletons, frame)
                for skeleton_id, features in feature_tracker.extract():
                    our_model.predict(features)


def compute_label(frame_id: str, classes: list[dict]) -> str:
    frame_id = int(frame_id.removesuffix(".jpg"))

    i_class = 0
    start_frame = classes[0]["start_frame"]
    while frame_id < start_frame:
        i_class += 1
        start_frame = classes[i_class]["start_frame"]

    if frame_id < classes[i_class]["end_frame"]:
        return classes[i_class]["classification"]

    return "neutral"


available_classes = ['kicking', 'punching', 'pushing', 'neutral']
def label_to_int(label: str) -> int:
    return available_classes.index(label)

def get_angle_list(number_of_angles: int) -> list[tuple[int, int, int]]:
    if number_of_angles == 0:
        return []

    if number_of_angles == 4:
        return skeletonization.BK.BASIC_ANGLE_LIST

    if number_of_angles == 8:
        return skeletonization.BK.MEDIUM_ANGLE_LIST
