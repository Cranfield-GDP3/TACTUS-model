'''
This script defines the functions for reading skeletons information:

* def load_skeleton_data
    
    Load data from `skeletons_data.txt`.

'''

import numpy as np
import cv2
import os
import sys
import simplejson
from sklearn.preprocessing import OneHotEncoder


# Input format:
#[cn_video, cnt_frame_inside_video, cnt_frame_global, img_action_label, filepath, keypoints]

LEN_IMG_INFO = 5
LEN_SKELETON_XY = 13*2
NaN = 0  
if True:  
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import tactus_model.utils.lib_helpers as lib_commons


def par(path):  
    return ROOT + path if (path and path[0] != "/") else path

cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
DATA_AUGMENT = int(cfg_all["features"]["add_augmentation"]) 

# Function to load the skeletons_data

def load_skeleton_data(filepath, classes):
    ''' Load training data from skeletons_info.txt.
    Some notations:
        N: number of valid data.
        P: feature dimension. Here P=36.
        C: number of classes.
    Arguments:
        filepath {str}: file path of `skeletons_info.txt`, which stores the skeletons and labels.
    Returns:
        X: {np.array, shape=NxP}:           Skeleton data (feature) of each valid image.
        Y: {list of int, len=N}:            Label of each valid image.
        video_indices {list of int, len=N}:  The video index of which the image belongs to.
        classes {list of string, len=C}:    The classes of all actions.
    '''
  
    idx_label = {c: i for i, c in enumerate(classes)}
    
    with open(filepath, 'r') as f:
       
        # Load data
        dataset = simplejson.load(f)
       
        # Remove invalid data
        def valid_data(row):
            return row[0] != 0
        dataset = [row for i, row in enumerate(dataset) if valid_data(row)]

        X = np.array([row[LEN_IMG_INFO:LEN_IMG_INFO+LEN_SKELETON_XY]
                      for row in dataset])
    
        # row[0] is the video index of the image
        if not DATA_AUGMENT:
            video_indices = [row[0] for row in dataset] 
        else:
            video_indices = [row[1] for row in dataset] 
            
        Y_str = [row[3] for row in dataset] #for labels

        Y = [idx_label[label] for label in Y_str]

        # Remove data with missing upper body joints
        if 0:
            valid_indices = _get_skeletons_with_complete_upper_body(X, NaN)
            X = X[valid_indices, :]
            Y = [Y[i] for i in valid_indices]
            video_indices = [video_indices[i] for i in valid_indices]
            print("Num samples after removal = ", len(Y))

        # properties of input data
        N = len(Y)
        P = len(X[0])
        C = len(classes)
        print(f"\nNumber of samples = {N} \n"
              f"Raw feature length = {P} \n"
              f"Number of classes = {C}")
        print(f"Classes: {classes}")
        return X, Y, video_indices

    raise RuntimeError("Failed to load skeletons txt: " + filepath)


def _get_skeletons_with_complete_upper_body(X, NaN=0):
    ''' 
    Find skeletons with valid upper body joints 
    Return the indices of these skeletons.
   '''

    left_idx, right_idx = 0, 13 * 2  # TODO: will be redundant in case of YOLOv7

    def is_valid(x):
        return len(np.where(x[left_idx:right_idx] == NaN)[0]) == 0
    valid_indices = [i for i, x in enumerate(X) if is_valid(x)]
    return valid_indices

