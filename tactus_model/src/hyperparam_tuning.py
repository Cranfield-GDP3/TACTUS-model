import json
from pathlib import Path
import numpy as np
from tactus_data import skeletonization, data_augment
from tactus_data.datasets.ut_interaction import data_split
from tactus_model.utils.tracker import FeatureTracker
from tactus_model.utils.classifier import Classifier

AVAILABLE_CLASSES = ['kicking', 'punching', 'pushing', 'neutral']


DATA_AUGMENT_GRIDS = {
    # "BIG_GRID": {
    #     "noise_amplitude": [0, 3],
    #     "horizontal_flip": [True, False],
    #     "rotation_y": [-30, -20, -10, 0, 10, 20, 30],
    #     "rotation_z": [-10, 0, 10],
    #     "rotation_x": [-30, -20, -10, 0, 10, 20, 30],
    #     "scale_x": [0.9, 1.1],
    #     "scale_y": [0.9, 1.1],
    # },
    # "DEFAULT_GRID": {
    #     "noise_amplitude": [0, 3],
    #     "horizontal_flip": [True, False],
    #     "rotation_y": [-30, 0, 30],
    #     "rotation_z": [-10, 0, 10],
    #     "rotation_x": [-30, 0, 30],
    #     "scale_x": [0.8, 1, 1.2],
    #     "scale_y": [0.8, 1, 1.2],
    # },
    # "MED_GRID": {
    #     "noise_amplitude": [0, 3],
    #     "horizontal_flip": [True, False],
    #     "rotation_y": [-30, -20, -10, 0, 10, 20, 30],
    #     "rotation_z": [-10, 0, 10],
    #     "rotation_x": [-30, -20, -10, 0, 10, 20, 30],
    # },
    # "SMALL_GRID": {
    #     "noise_amplitude": [0, 2, 4],
    #     "horizontal_flip": [True, False],
    #     "rotation_y": [-30, 0, 30],
    #     "rotation_z": [-10, 0, 10],
    #     "rotation_x": [-30, 0, 30],
    # },
    "Flip": {"horizontal_flip": [True, False]},
    "SMALLER_GRID": [  # grid size: 32
        # {
        #     "horizontal_flip": [True, False],
        #     "scale_x": [1, 1.2],
        #     "scale_y": [0.8, 1],
        # },
        {
            "horizontal_flip": [True, False],
            "scale_x": [0.8],
            "scale_y": [1.2, 1],
        },
        # {
        #     "horizontal_flip": [True, False],
        #     "scale_x": [1],
        #     "scale_y": [1.2],
        # },
        # {
        #     "horizontal_flip": [True, False],
        #     "rotation_y": np.linspace(-30, 30, 3),
        #     "rotation_x": np.linspace(-20, 20, 3),
        # },
    ]
}


TRACKER_GRID = {  # grid size: 12
    "window_size": [3, 5, 9, 15],  # impact velocity
    "number_of_angles": [0, 4, 8],
}


CLASSIFIER_HYPERPARAMS = {
    # "lstm": {  # grid size: 72
    #     "batch_size": [64, 256],
    #     "epochs": [100],
    #     "neurons": [50, 100, 250, 500],
    #     "activation": ["tanh", "relu", "sigmoid"],
    #     "features_size": ["computed later"],
    #     "dropout_layer": [0, 0.2, 0.4],
    # },
    "SVC": {
        "C": [0.5, 1, 2],
        "degree": [3, 5, 7],
    },
}


def get_classifier():
    for classifier_name, hyperparams_grid in CLASSIFIER_HYPERPARAMS.items():
        for hyperparams in data_augment.ParameterGrid(hyperparams_grid):
            yield Classifier(classifier_name, hyperparams), classifier_name, hyperparams


def get_augment_grid():
    for _, augment_grid in DATA_AUGMENT_GRIDS.items():
        yield augment_grid


def get_tracker_grid():
    for grid in data_augment.ParameterGrid(TRACKER_GRID):
        yield grid


def train(fps: int = 10):
    # cant use a generator here because we use this multiple times
    train_videos, _, test_videos = data_split(Path("data/processed/ut_interaction/"), (85, 0, 15))

    count = 0

    # delete every augmentation data. Necessary in case the new
    # augmentation is smaller than the former one
    for video_path in train_videos + test_videos:
        for augmented_data_path in video_path.rglob("*_augment_*"):
            augmented_data_path.unlink()

    for augment_grid in get_augment_grid():
        # augments data and saves it in files
        print("augments data with: ", augment_grid)
        for video_path in train_videos:
            original_data_path = video_path / f"{fps}fps" / "yolov7.json"
            data_augment.grid_augment(original_data_path, augment_grid)

        # compute features
        for tracker_grid in get_tracker_grid():
            print("compute features with: ", tracker_grid)
            angle_list = get_angle_list(tracker_grid["number_of_angles"])
            window_size = tracker_grid["window_size"]

            X, Y = generate_features(train_videos, fps, window_size, angle_list)
            X_test, Y_test = generate_features(test_videos, fps, window_size, angle_list)

            save_file = {}
            save_file["augment_grid"] = augment_grid
            save_file["tracker_grid"] = tracker_grid
            for classifier, classifier_name, hyperparams in get_classifier():
                print("fit classifier: ", classifier_name, " - ", hyperparams)
                classifier.fit(X, Y)

                save_file["classifier_name"] = classifier_name
                save_file["hyperparams"] = hyperparams
                save_file["y_pred_train"] = classifier.predict(X)
                save_file["y_pred_proba_train"] = classifier.predict_proba(X)
                save_file["y_true_train"] = Y
                save_file["y_pred_test"] = classifier.predict(X_test)
                save_file["y_pred_proba_test"] = classifier.predict_proba(X_test)
                save_file["y_true_test"] = Y_test

                filename = Path(f"data/models/evaluation/{count}.json")
                json.dump(save_file, filename.open(w))

                count += 1


def generate_features(videos: list[Path], fps: int, window_size: int, angle_list: list):
    X = []
    Y = []

    for video_path in videos:
        label_path = video_path / (f"{video_path.stem}.label.json")
        labels = json.load(label_path.open())

        for augmented_data_path in video_path.glob(f"{fps}fps/*_augment_*.json"):
            augmented_data = json.load(augmented_data_path.open())

            video_features, video_labels = feature_from_video(augmented_data, labels, window_size, angle_list)

            X.extend(video_features)
            Y.extend(video_labels)

    return X, Y


def feature_from_video(augmented_data: dict,
                       labels: dict,
                       window_size: int,
                       angle_list: list):
    feature_tracker = FeatureTracker(window_size=window_size, angles_to_compute=angle_list)

    video_features = []
    video_labels = []

    offender_id = labels["offender"][0]

    i_label = 0
    i_frame = max(labels["classes"][i_label]["start_frame"] - window_size, 0)
    while i_frame < len(augmented_data["frames"]):
        frame = augmented_data["frames"][i_frame]

        for skeleton in frame["skeletons"]:
            # do not deal with untracked skeletons
            if "id_stupid" not in skeleton:
                continue

            feature_tracker.update_rolling_window(skeleton["id_stupid"],
                                                  skeleton,
                                                  has_head=False,
                                                  has_confidence=False)

        for skeleton_id, (success, features) in feature_tracker.extract():
            if not success:
                continue

            if skeleton_id == offender_id:
                label = compute_label(frame["frame_id"], labels["classes"], i_label)
            else:
                label = "neutral"

            video_features.append(features)
            video_labels.append(label_to_int(label))

        # jump to the next action
        if i_label >= labels["classes"][i_label]["end_frame"]:
            if len(labels["classes"]) > i_label:
                i_label += 1
                i_frame = labels["classes"][i_label]["start_frame"] - window_size
                feature_tracker.reset_rolling_windows()
            else:
                break
        else:
            i_frame += 1

    return video_features, video_labels


def compute_label(frame_id: str, classes: list, i_label: int) -> str:
    frame_id = int(frame_id.removesuffix(".jpg"))

    if classes[i_label]["start_frame"] < frame_id < classes[i_label]["end_frame"]:
        return classes[i_label]["classification"]

    return "neutral"


def label_to_int(label: str) -> int:
    return AVAILABLE_CLASSES.index(label)


def get_angle_list(number_of_angles: int) -> list[tuple[int, int, int]]:
    if number_of_angles == 0:
        return []

    if number_of_angles == 4:
        return skeletonization.BK.BASIC_ANGLE_LIST

    if number_of_angles == 8:
        return skeletonization.BK.MEDIUM_ANGLE_LIST
