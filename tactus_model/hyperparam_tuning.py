from typing import List, Dict, Tuple, Generator
import json
from pathlib import Path
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tactus_data import data_augment, SMALL_ANGLES_LIST, MEDIUM_ANGLES_LIST
from tactus_data.datasets.ut_interaction import data_split
from tactus_data import Skeleton

from tactus_model.utils.tracker import FeatureTracker
from tactus_model.utils.classifier import Classifier, label_to_int

DATA_AUGMENT_GRIDS = {
    "FLIP": {"horizontal_flip": [True, False]},
}

TRACKER_GRID = {
    "window_size": [9],
    "number_of_angles": [0],
}

CLASSIFIER_HYPERPARAMS = {
    "TorchMLP": {
        "batch_size": [256],
        "num_epochs": [30],
        "hidden_layer_sizes": [{"layer_sizes": [256, 128, 16], "dropout": [0.4, 0.2, 0.1]},
                               {"layer_sizes": [1024, 512, 128, 16], "dropout": [0.5, 0.3, 0.2, 0.1]},
                               {"layer_sizes": [512, 128, 64], "dropout": [0.2, 0.2, 0.1]},
                               ],
        "activation": ['ReLU']
    },
}


def get_classifier(
        classifier_grids: Dict[str, Dict] = None,
) -> Generator[Tuple[Classifier, str, dict], None, None]:
    """
    get a classifier instance with a set of hyperparametres found in
    the grid.

    Yields
    ------
    Generator[Tuple[Classifier, str, dict], None, None]
        a tuple with
        - the classifier initialised with the hyperparametres
        - the classifier name
        - the hyperparamtres used on the classifier
    """
    if classifier_grids is None:
        classifier_grids = CLASSIFIER_HYPERPARAMS

    for classifier_name, hyperparams_grid in classifier_grids.items():
        for hyperparams in data_augment.ParameterGrid(hyperparams_grid):
            hyperparams["dropout_layers_sizes"] = hyperparams["hidden_layer_sizes"]["dropout"]
            hyperparams["hidden_layer_sizes"] = hyperparams["hidden_layer_sizes"]["layer_sizes"]
            yield Classifier(classifier_name, hyperparams)


def get_augment_grid(
        augment_grids: Dict[str, Dict] = None,
) -> Generator[dict, None, None]:
    """
    get a data augmentation grid from augment_grids.

    Yields
    ------
    Generator[dict, None, None]
    """
    if augment_grids is None:
        augment_grids = DATA_AUGMENT_GRIDS

    for _, augment_grid in augment_grids.items():
        yield augment_grid


def get_tracker_params(
        tracker_grid: Dict[str, List] = None,
) -> Generator[dict, None, None]:
    """
    get a tracker parametres (with window size and number of angle) from
    the grid.

    Yields
    ------
    Generator[dict, None, None]
    """
    if tracker_grid is None:
        tracker_grid = TRACKER_GRID

    for grid in data_augment.ParameterGrid(tracker_grid):
        yield grid


def sample_data(X: List, Y: List, sampler: str):
    """
    Over/Under sample the data if the flag is set to the corresponding value, does nothing if flag = X
    Args:
        X: Data for the classifier
        Y: Label for the classifier
        sampler: Flag chosing the sample technique

    Returns:
        The sampled data
    """
    if sampler == "RUS":
        sampler = RandomUnderSampler()
    elif sampler == "SMOTE":
        sampler = SMOTE()
    else:
        return X, Y
    return sampler.fit_resample(X, Y)


def train_grid_search(
        fps: int = 10,
        augment_grids: Dict[str, Dict] = None,
        tracker_grid: Dict[str, List] = None,
        classifier_grids: Dict[str, Dict] = None,
        sampler: str = "RUS"
):
    """
    launch the training process

    Parameters
    ----------
    fps : int, optional
        the fps to train on, by default 10
    """
    # cant use a generator here because we use this multiple times
    train_videos, _, test_videos = data_split(Path("data/processed/ut_interaction/"), (85, 0, 15))

    model_id = 0
    for augment_grid in get_augment_grid(augment_grids):
        # augments data and saves it in files
        print("augments data with: ", augment_grid)
        delete_data_augment(train_videos + test_videos, fps)

        for video_path in train_videos:
            original_data_path = video_path / f"{fps}fps" / "yolov7.json"
            data_augment.grid_augment(original_data_path, augment_grid)

        for video_path in test_videos:
            original_data_path = video_path / f"{fps}fps" / "yolov7.json"
            data_augment.grid_augment(original_data_path, {})

        # compute features
        for tracker_params in get_tracker_params(tracker_grid):
            print("compute features with: ", tracker_params)
            angle_list = get_angle_list(tracker_params["number_of_angles"])
            window_size = tracker_params["window_size"]

            X, Y = generate_features(train_videos, fps, window_size, angle_list)
            X_test, Y_test = generate_features(test_videos, fps, window_size, angle_list)
            X,Y = sample_data(X, Y, sampler)

            for classifier in get_classifier(classifier_grids):
                print("fit classifier: ", classifier.name, " with ", classifier.hyperparams)
                classifier.fit(X, Y)
                classifier.window_size = window_size
                classifier.fps = fps
                classifier.angles_to_compute = angle_list
                classifier.save(Path(f"data/models/pickle/{model_id}.pickle"))

                save_model_evaluation(model_id, classifier, augment_grid,
                                      tracker_params, X, Y, X_test, Y_test)

                model_id += 1


def save_model_evaluation(model_id, classifier, augment_grid, tracker_params, X, Y, X_test, Y_test):
    """
    predict the train and the test dataset to measure the performance
    of the model later on.
    """
    eval_dict = {}

    eval_dict["augment_grid"] = augment_grid
    eval_dict["tracker_params"] = tracker_params

    eval_dict["classifier_name"] = classifier.name
    eval_dict["hyperparams"] = classifier.hyperparams

    eval_dict["y_pred_train"] = classifier.predict(X)
    eval_dict["y_true_train"] = Y

    eval_dict["y_pred_test"] = classifier.predict(X_test)
    eval_dict["y_true_test"] = Y_test

    filename = Path(f"data/models/evaluation/{model_id}.json")
    filename.parent.mkdir(parents=True, exist_ok=True)
    json.dump(eval_dict, filename.open(mode="w", encoding="utf8"))


def generate_features(videos: List[Path], fps: int, window_size: int, angle_list: List):
    """
    generates features for a list of video directories. The directories
    must follow the following structure:
    `video_name -> label.json`
    `video_name -> xxfps -> *_augment_*.json`

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


def feature_from_video(
    formatted_json: Dict,
    labels: Dict,
    window_size: int,
    angle_list: List
):
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
    i_frame = get_i_frame(labels["classes"][i_label]["start_frame"], formatted_json["frames"])

    while i_frame < len(formatted_json["frames"]):
        frame = formatted_json["frames"][i_frame]

        for skeleton in frame["skeletons"]:
            # do not deal with untracked skeletons
            if "id_stupid" in skeleton:
                skeleton_id = skeleton["id_stupid"]

                keypoints = np.array(skeleton["keypoints"]).reshape((-1, 2))
                feature_tracker.update_rolling_window(skeleton_id, Skeleton(keypoints=keypoints))

                if feature_tracker[skeleton_id].is_complete():
                    features = feature_tracker[skeleton_id].get_features()

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
    frame_id = frame_id_to_int(frame_id)

    if classes[i_label]["start_frame"] < frame_id < classes[i_label]["end_frame"]:
        return classes[i_label]["classification"]

    return "neutral"


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
        return SMALL_ANGLES_LIST

    if number_of_angles == 8:
        return MEDIUM_ANGLES_LIST


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


def get_i_frame(target_frame_id: int,
                frames: List):
    """
    convert a frame_id to its index in a list.

    Parameters
    ----------
    starting_frame: int
        number of the starting frame.
    frames: List
        list containing frame info.
    """
    frame_index = 0
    current_frame_id = frame_id_to_int(frames[frame_index]["frame_id"])

    while current_frame_id < target_frame_id:
        current_frame_id = frame_id_to_int(frames[frame_index]["frame_id"])
        frame_index += 1
    return frame_index


def frame_id_to_int(frame_id: str) -> int:
    """
    convert the id of a frame (003.jpg) to an int (3).

    removesuffix is not available in python 3.8
    """
    return int(frame_id[:-4])


if __name__ == "__main__":
    train_grid_search()
