from typing import List, Dict, Tuple, Generator
import json
import numpy as np
from pathlib import Path
from tactus_data import skeletonization, data_augment
from tactus_data.datasets.ut_interaction import data_split
from tactus_model.utils.tracker import FeatureTracker
from tactus_model.utils.classifier import Classifier

AVAILABLE_CLASSES = ['kicking', 'punching', 'pushing', 'neutral']



DATA_AUGMENT_GRIDS = {
    "FLIP": {"horizontal_flip": [True]},
    "SMALL_GRID": {
        "noise_amplitude": [0, 2, 4],
        "horizontal_flip": [True, False],
        "rotation_y": [-30, 0, 30],
        "rotation_z": [-10, 0, 10],
        "rotation_x": [-30, 0, 30],
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
    "BIG_GRID": {
        "noise_amplitude": [0, 3],
        "horizontal_flip": [True, False],
        "rotation_y": [-30, -20, -10, 0, 10, 20, 30],
        "rotation_z": [-10, 0, 10],
        "rotation_x": [-30, -20, -10, 0, 10, 20, 30],
        "scale_x": [0.9, 1.1],
        "scale_y": [0.9, 1.1],
    },

}


TRACKER_GRID = {  # grid size: 6
    "window_size": [3, 5, 9],  # impact velocity
    "number_of_angles": [0, 8],
}


CLASSIFIER_HYPERPARAMS = { #grid size : 24
    "MLPClassifier": {
        "batch_size": [256],
        "max_iter": [300],
        #"loss_function": ["SparseCategoricalCrossEntropy"],
        "hidden_layer_sizes": [(256,128,16,),(1024,512,128,16,),(512,64,32,)],
        "activation": ['tanh','relu'],
        "alpha": [0.05,0.1,1],
        "solver": ['adam'],
        "learning_rate": ['adaptive'],
        #"dropout_layer": [0,0.2,0.4],
        "random_state": [42],
    },
}


def get_classifier() -> Generator[Tuple[Classifier, str, dict], None, None]:
    """
    get a classifier instance with a set of hyperparametres found in
    the grid CLASSIFIER_HYPERPARAMS.

    Yields
    ------
    Generator[Tuple[Classifier, str, dict], None, None]
        a tuple with
        - the classifier initialised with the hyperparametres
        - the classifier name
        - the hyperparamtres used on the classifier
    """
    for classifier_name, hyperparams_grid in CLASSIFIER_HYPERPARAMS.items():
        for hyperparams in data_augment.ParameterGrid(hyperparams_grid):
            yield Classifier(classifier_name, hyperparams), classifier_name, hyperparams


def get_augment_grid() -> Generator[dict, None, None]:
    """
    get a data augmentation grid from DATA_AUGMENT_GRIDS.

    Yields
    ------
    Generator[dict, None, None]
    """
    for _, augment_grid in DATA_AUGMENT_GRIDS.items():
        yield augment_grid


def get_tracker_grid() -> Generator[dict, None, None]:
    """
    get a tracker grid (with window size and number of angle) from
    TRACKER_GRID.

    Yields
    ------
    Generator[dict, None, None]
    """
    for grid in data_augment.ParameterGrid(TRACKER_GRID):
        yield grid


def train(fps: int = 10):
    """
    launch the training process

    Parameters
    ----------
    fps : int, optional
        the fps to train on, by default 10
    """
    # cant use a generator here because we use this multiple times
    train_videos, _, test_videos = data_split(Path("data/processed/ut_interaction/"), (85, 0, 15))

    count = 0
    for augment_grid in get_augment_grid():
        # augments data and saves it in files
        print("augments data with: ", augment_grid)
        delete_data_augment(train_videos + test_videos, fps)

        for video_path in train_videos + test_videos:
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
                loss_history = classifier.fit(X, Y).loss_curve_

                save_file["classifier_name"] = classifier_name
                save_file["hyperparams"] = hyperparams
                save_file["y_pred_train"] = classifier.predict(X).tolist()
                save_file["y_true_train"] = Y
                save_file["y_pred_test"] = classifier.predict(X_test).tolist()
                save_file["y_true_test"] = Y_test
                save_file["loss_history"] = loss_history
                filename = Path(f"data/models/evaluation/{count}.json")
                json.dump(save_file, filename.open(mode="w"))


                classifier.save(Path(f"data/models/pickle/{count}.json"))

                count += 1


def generate_features(videos: List[Path], fps: int, window_size: int, angle_list: List):
    """
    generates features for a list of video directories. The directories
    must follow the following structure:
    `video_name -> label.json`
    `video_name -> xxfps -> data.json`

    Parameters
    ----------
    videos : List[Path]
        list of the video directories.
    fps : int
        the fps to generate features with.
    window_size : int
        the size of the rolling window.
    angle_list : List
        the list of the angles to compute.

    Returns
    -------
    (X, Y)
        return a tuple with the features and the labels.
    """
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


def feature_from_video(formatted_json: Dict,
                       labels: Dict,
                       window_size: int,
                       angle_list: List):
    """
    Compute features and true labels from a `json video` file.

    Parameters
    ----------
    formatted_json : Dict
        video json file containing the list of frames and their
        skeletons.
    labels : Dict
        the content of the video label file.
    window_size : int
        the size of the rolling window.
    angle_list : List
        the list of angles to compute in the features.

    Returns
    -------
    (X, Y)
        return a tuple with the features and the labels.
    """
    feature_tracker = FeatureTracker(window_size=window_size, angles_to_compute=angle_list)

    video_features = []
    video_labels = []

    offender_id = labels["offender"][0]

    i_label = 0
    i_frame = get_i_frame(max(labels["classes"][i_label]["start_frame"] - window_size, 0),
                          formatted_json["frames"])

    while i_frame < len(formatted_json["frames"]):
        frame = formatted_json["frames"][i_frame]

        for skeleton in frame["skeletons"]:
            # do not deal with untracked skeletons
            if "id_stupid" in skeleton:
                feature_tracker.update_rolling_window(
                    skeleton["id_stupid"], skeleton,
                    has_head=False, has_confidence=False)

        for skeleton_id, (success, features) in feature_tracker.extract():
            if success:
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


def compute_label(frame_id: str, classes: List[Dict], i_label: int) -> str:
    """
    compute the label of the offender for a given frame.

    Parameters
    ----------
    frame_id : str
        the frame id. Usually "xxx.jpg".
    classes : list
        list of labels `classes`. It must have `end_frame`,
        `end_frame`, `classification` keys.
    i_label : int
        the index of the current class.

    Returns
    -------
    str
        the corresponding label.
    """
    frame_id = int(frame_id[:-4]) #removesuffix not in 3.8

    if classes[i_label]["start_frame"] < frame_id < classes[i_label]["end_frame"]:
        return classes[i_label]["classification"]

    return "neutral"


def label_to_int(label: str) -> int:
    """transform a label to its corresponding integer."""
    return AVAILABLE_CLASSES.index(label)


def get_angle_list(number_of_angles: int) -> List[Tuple[int, int, int]]:
    """
    get predefined list of angles to compute from the number
    of angles we would like.

    Parameters
    ----------
    number_of_angles : int
        number of angles to compute. Must be 0, 4 or 8.

    Returns
    -------
    list[tuple[int, int, int]]
        the list of angles to compute.
    """
    if number_of_angles == 0:
        return []

    if number_of_angles == 4:
        return skeletonization.BK.BASIC_ANGLE_LIST

    if number_of_angles == 8:
        return skeletonization.BK.MEDIUM_ANGLE_LIST


def delete_data_augment(video_paths: List[Path], fps: int):
    """
    delete every augmentation data. Necessary in case the new
    augmentation is smaller than the former one.

    Parameters
    ----------
    video_paths : List[Path]
        list of the video directories.
    fps : int
        the fps of the data augmentation to delete.
    """
    for video_path in video_paths:
        for augmented_data_path in video_path.glob(f"{fps}fps/*_augment_*"):
            augmented_data_path.unlink()

def get_i_frame(starting_frame: int,
                frames: Dict):
    current_frame = -100000
    counter = 0

    while current_frame < starting_frame:
        current_frame = int(frames[counter]["frame_id"][:-4])
        counter+=1
    return counter


train()