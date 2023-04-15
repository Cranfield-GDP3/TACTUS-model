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
from collections import Sequence
import copy
import time
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track
from tactus_data import retracker
from tactus_data import SkeletonRollingWindow


class FeatureTracker:
    """
    High level interface with rolling windows of skeletons and their
    tracking with deepsort.
    """
    def __init__(self,
                 deepsort_tracker: DeepSort = None,
                 window_size: int = 5,
                 angles_to_compute: List[Tuple[int, int, int]] = None
                 ):
        self.window_size = window_size
        self.angles_to_compute = angles_to_compute

        self.rolling_windows: dict[int, SkeletonRollingWindow]
        self.reset_rolling_windows()

        if deepsort_tracker is None:
            self.deepsort = DeepSort(n_init=3, max_age=5)
        else:
            self.deepsort = deepsort_tracker
        self.tracks_to_del = []

    def reset_rolling_windows(self):
        self.rolling_windows = {}

    def track_skeletons(self, skeletons, frame: np.ndarray):
        """run the tracker on each skeleton and update their rolling
        window"""
        tracks: list[Track]
        tracks = retracker.deepsort_track_frame(self.deepsort, frame, skeletons, new_version=True)
        self.tracks_to_del = copy.deepcopy(self.deepsort.tracker.del_tracks_ids)

        for i, track in enumerate(tracks):
            if track.is_confirmed():
                self.update_rolling_window(track.track_id, skeletons[i])

        for track_id in self.tracks_to_del:
            self.delete_track_id(track_id)

    def update_rolling_window(self, track_id: int, skeleton: dict, has_head: bool = True, has_confidence: bool = True):
        """update a SkeletonRollingWindow from its ID"""
        if track_id not in self.rolling_windows:
            self.rolling_windows[track_id] = SkeletonRollingWindow(self.window_size, self.angles_to_compute)

        self.rolling_windows[track_id].add_skeleton(skeleton, has_head, has_confidence)

    def delete_track_id(self, track_id: int):
        """delete a SkeletonRollingWindow from its ID"""
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


class PredTracker:
    """
    save the non-neutral predictions for each skeleton still present
    on the stream.
    """
    def __init__(self):
        self.tracker: Dict[int, Dict]

    def add_pred(self, track_id: int, label: str, bbx: Tuple[int, int, int, int]):
        """starts the tracking of a person from a prediction label"""
        pred_tracker_info = {
            "label": label,
            "timestamp": time.time(),
            "original_bbx": bbx
        }

        self.tracker[track_id] = pred_tracker_info

    def remove_pred_track(self, track_ids: Union[List[int], int]):
        """removes the track of a person"""
        if not isinstance(track_ids, Sequence):
            track_ids = [track_ids]

        for track_id in track_ids:
            del self[track_id]

    def __getattr__(self, __index: int) -> Dict[int, Dict]:
        return self.tracker[__index]

    def __delattr__(self, __index: int):
        del self.tracker[__index]
