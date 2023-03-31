#!/usr/bin/env python
# coding: utf-8

'''
Input:
    JSON files for label and keypoints.
Output:
    `output2.txt`. The filepath is `DST_ALL_SKELETONS_TXT`.
'''

import numpy as np
import simplejson
import collections
import json

if True:  
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_helpers as lib_commons


def par(path):  
    return ROOT + path if (path and path[0] != "/") else path

# Configuration settings


cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["combine_skeletons_info.py"]

CLASSES = np.array(cfg_all["classes"])

SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

SRC_DETECTED_SKELETONS_FOLDER = par(cfg["input"]["detected_skeletons_folder"])
DST_ALL_SKELETONS_TXT = par(cfg["output"]["all_skeletons_txt"])

IDX_ACTION_LABEL=3
KEYPOINTS_FOLDER = "/10fps"
KEYPOINTS_FILE = "alphapose_2d.json"
JSON = ".json"

# JSON METADATA
OFFENDER = "offender"
FRAME_START = "start_frame"
FRAME_END = "end_frame"
_CLASS = "classes"
ACTION_CLASS = "classification"
FRAME = "frames"
HUMANS = "skeletons"
TRACKING_ID = "id_stupid"
FRAME_ID = "frame_id"
KEYPOINTS = "keypoints"

def get_all_skeletons_info():
  #with open(DST_ALL_SKELETONS_TXT, "w") as f:
  # TODO: ADD TQDM for next commit
  print("Extracting information ...")
  training_dataframe = []
  all_valid_frames = 0 
  for folder_num, file_ in enumerate(os.listdir(SRC_DETECTED_SKELETONS_FOLDER)):

    '''
    extract: action label, start and stop frame 
    '''
    for label_file in os.listdir(SRC_DETECTED_SKELETONS_FOLDER+"/"+file_):
      if label_file.endswith(JSON):       
        with open(os.path.join(SRC_DETECTED_SKELETONS_FOLDER+"/"+file_, label_file)) as labels_:
          label_data = json.load(labels_)
          label_= label_data[_CLASS][0][ACTION_CLASS]
          offender_ = label_data[OFFENDER]
          if offender_[0] == 0:
            offender_[0]+=1
          start_ = label_data[_CLASS][0][FRAME_START]
          end_ = label_data[_CLASS][0][FRAME_END]

    for filename in os.listdir(SRC_DETECTED_SKELETONS_FOLDER +"/"+file_+ KEYPOINTS_FOLDER):  
        
      valid_frame =0
      if filename == KEYPOINTS_FILE:
        with open(os.path.join(SRC_DETECTED_SKELETONS_FOLDER+"/"+file_+ KEYPOINTS_FOLDER , filename)) as json_file:
          data = json.load(json_file)
          for frames in data[FRAME]:
            for points in frames[HUMANS]:
              if points[TRACKING_ID] == offender_[0]:                
                if int(frames[FRAME_ID][:-4]) >=start_ and int(frames[FRAME_ID][:-4]) <= end_:                  
                  valid_frame+=1
                  all_valid_frames+=1
                  # extract keypoints
                  key_frame = points[KEYPOINTS]
                  key_frame_misc = [folder_num+1, valid_frame, all_valid_frames, label_, frames[FRAME_ID]]                 
                  training_dataframe.append(key_frame_misc + [points[KEYPOINTS][i] for i in range(len(points[KEYPOINTS])) if (i+1) % 3 != 0])
  return training_dataframe
                 
  

def main():
  training_dataframe = get_all_skeletons_info()
  print("Writing information to skeletons_data.txt ...")
  all_skeletons = []
  labels_cnt = collections.defaultdict(int)
  for i in training_dataframe:
    # Read skeletons from a txt
    skeletons = i
    #print(skeletons)
    if not skeletons:  # If empty, discard this image.
      continue
    skeleton = skeletons
        
    label = skeleton[IDX_ACTION_LABEL]
    if label not in CLASSES:  # If invalid label, discard this image.
      continue
    labels_cnt[label] += 1
  
    # Push to result
    all_skeletons.append(skeleton)

    # -- Save to txt
    with open(DST_ALL_SKELETONS_TXT, 'w') as f:
      simplejson.dump(all_skeletons, f)

  print(f"There are {len(all_skeletons)} skeleton data.")
  
  print("Number of each action: ")
  for label in CLASSES:
      print(f"    {label}: {labels_cnt[label]}")

if __name__ == "__main__":
    main()