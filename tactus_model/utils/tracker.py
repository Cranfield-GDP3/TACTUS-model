"""
keep a rolling window for every skeleton on a frame. If the skeletons
disappear, its rolling window is going to be deleted.

they are all tracked inside a dictionnary that has the tracking id
as a key e.g.
{
    1: SkeletonRollingWindow(...),
    2: SkeletonRollingWindow(...),
    6: SkeletonRollingWindow(...),
    12: SkeletonRollingWindow(...),
}
"""
from typing import Union, Generator, Tuple, List, Dict
from collections.abc import Sequence
import time
import numpy as np
from tactus_data import SkeletonRollingWindow, Skeleton


class FeatureTracker:
    """
    High level interface with rolling windows of skeletons and their
    tracking with deepsort.
    """
    def __init__(self,
                 window_size: int = 5,
                 angles_to_compute: List[Tuple[int, int, int]] = None
                 ):
        self.window_size = window_size
        self.angles_to_compute = angles_to_compute

        self.rolling_windows: dict[int, SkeletonRollingWindow]
        self.reset_rolling_windows()

        self.tracks_to_del = []

    def reset_rolling_windows(self):
        self.rolling_windows = {}

    def update_rolling_window(self, track_id: int, skeleton: dict):
        """update a SkeletonRollingWindow from its ID"""
        if track_id not in self.rolling_windows:
            self.rolling_windows[track_id] = SkeletonRollingWindow(self.window_size, self.angles_to_compute)

        self.rolling_windows[track_id].add_skeleton(skeleton)

    def duplicate_last_entry(self, track_id: int, new_bbox_lbrt: Tuple[float, float, float, float]) -> Skeleton:
        """
        duplicate the last entry of a rolling window. That is useful
        to avoid gaps in the prediction.
        """
        return self.rolling_windows[track_id].duplicate_last_entry(new_bbox_lbrt)

    def delete_track_id(self, track_id: int):
        """
        delete a SkeletonRollingWindow from its ID.

        DeepSort can have unconfirmed track that are not taken into
        account in FeatureTracker. To avoid an error, we delete
        the index only if it exists.
        """
        if track_id in self.rolling_windows:
            del self.rolling_windows[track_id]

    def extract(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract features from each SkeletonRollingWindow

        Yields
        ------
        Generator[int, np.ndarray]
            yields (track_id, features) for each SkeletonRollingWindow
        """
        for track_id, rolling_window in self.rolling_windows.items():
            yield track_id, rolling_window.get_features()

    def __getitem__(self, __track_id: int) -> SkeletonRollingWindow:
        return self.rolling_windows[__track_id]


class PredTracker:
    """
    save the non-neutral predictions for each skeleton still present
    on the stream.
    """
    def __init__(self):
        self.tracker: Dict[int, Dict] = {}

    def init_track(self, track_id: int):
        """initialize the dictionary key"""
        self.tracker[track_id] = {"current_label": None,
                                  "label_history": []}

    def add_pred(self, track_id: int, label: str, bbx: Tuple[int, int, int, int]):
        """starts the tracking of a person from a prediction label"""
        if track_id not in self.tracker:
            self.init_track(track_id)

        self.tracker[track_id]["current_label"] = label
        self.tracker[track_id]["timestamp"] = time.time()
        self.tracker[track_id]["box"] = bbx

        if len(self.tracker[track_id]["label_history"]) > 0:
            if label != self.tracker[track_id]["label_history"][-1]:
                self.tracker[track_id]["label_history"].append(label)

    def delete_track_id(self, track_ids: Union[List[int], int]):
        """removes the track of a person"""
        if not isinstance(track_ids, Sequence):
            track_ids = [track_ids]

        for track_id in track_ids:
            if track_id in self.tracker:
                del self.tracker[track_id]

    def __contains__(self, __track_id: int):
        return __track_id in self.tracker

    def __getitem__(self, __track_id: int):
        return self.tracker[__track_id]
