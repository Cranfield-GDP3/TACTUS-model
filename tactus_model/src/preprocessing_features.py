#!/usr/bin/env python
# coding: utf-8


''' 
This is the source library for extracting features from skeleton points.
Following features are calculated:

1. Normalized keypoints
2. Joint velocities
3. Center of mass velocities


Input File: File with all skeletons information (skeletons_data.txt)
Output File: .CSV files for features and labels

'''

import numpy as np

if True:  
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import tactus_model.utils.lib_helpers as lib_commons
    from tactus_model.utils.lib_load_skeletons import load_skeleton_data
    from tactus_model.utils.lib_feature_extract import extract_multi_frame_features


def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path


# Configuration settings
cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["preprocessing_features.py"]

CLASSES = np.array(cfg_all["classes"])

# Window size
WINDOW_SIZE = int(cfg_all["features"]["window_size"]) 

# Input and output
SRC_ALL_SKELETONS_TXT = par(cfg["input"]["all_skeletons_txt"])
DST_PROCESSED_FEATURES = par(cfg["output"]["processed_features"])
DST_PROCESSED_FEATURES_LABELS = par(cfg["output"]["processed_features_labels"])


def process_skeletonpoints(X_, Y_, video_indices, classes):
    ''' Process features '''
    ADD_NOISE = False
    if ADD_NOISE:
        X1, Y1 = extract_multi_frame_features(
            X_, Y_, video_indices, WINDOW_SIZE, 
            is_adding_noise=True, is_print=True)
        X2, Y2 = extract_multi_frame_features(
            X_, Y_, video_indices, WINDOW_SIZE,
            is_adding_noise=False, is_print=True)
        X = np.vstack((X1, X2))
        Y = np.concatenate((Y1, Y2))
        return X, Y
    else:
        X, Y = extract_multi_frame_features(
            X_, Y_, video_indices, WINDOW_SIZE, 
            is_adding_noise=False, is_print=True)
        return X, Y


def main():
    ''' 
    Load skeleton data from `skeletons_data.txt`, process data, 
    and then save features and labels to .csv file.
    '''
    # Load the data from skeleton info file
    X_, Y_, video_indices = load_skeleton_data(SRC_ALL_SKELETONS_TXT, CLASSES)

    # Feature processing 
    X, Y = process_skeletonpoints(X_, Y_, video_indices, CLASSES)
    print(f"X.shape = {X.shape}, len(Y) = {len(Y)}")

    # Save data to CSV
    print("\nWriting features and label data ...")

    os.makedirs(os.path.dirname(DST_PROCESSED_FEATURES), exist_ok=True)
    os.makedirs(os.path.dirname(DST_PROCESSED_FEATURES_LABELS), exist_ok=True)

    np.savetxt(DST_PROCESSED_FEATURES, X, fmt="%.5f")
    print("Save features to: " + DST_PROCESSED_FEATURES)

    np.savetxt(DST_PROCESSED_FEATURES_LABELS, Y, fmt="%i")
    print("Save labels to: " + DST_PROCESSED_FEATURES_LABELS)


if __name__ == "__main__":
    main()
