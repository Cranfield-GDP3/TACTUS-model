#!/usr/bin/env python
# coding: utf-8


'''
NOTE: This function will be modified for new tracker Deepsort


Testing on
1. a video
2. folder of images
3. web camera.

Input:
    model: model/trained_classifier.pickle

Output:
    result video:    output/${video_name}/video.avi
    result skeleton: output/${video_name}/skeleton_res/XXXXX.txt
    visualization by cv2.imshow() in img_displayer
'''

'''
How to use this script?

1. Usage on video file:
python src/test.py \
    --model_path model/trained_classifier.pickle \
    --data_type video \
    --data_path data_test/exercise.avi \
    --output_folder output
    
2. Test on a folder of images:
python src/test.py \
    --model_path model/trained_classifier.pickle \
    --data_type folder \
    --data_path data_test/apple/ \
    --output_folder output

3. Test on web camera:
python src/test.py \
    --model_path model/trained_classifier.pickle \
    --data_type webcam \
    --data_path 0 \
    --output_folder output
    
'''


import numpy as np
import cv2
import argparse
import os
from PIL import Image
from pathlib import Path
if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
    ########### Note ##########
    # utils will be changed in the next commit
    from tactus_model.utils.lib_classifier import ClassifierOnlineTest
    from tactus_model.utils.lib_classifier import *
    import tactus_model.utils.lib_test_images_io as lib_images_io
    import tactus_model.utils.lib_plot as lib_plot
    import tactus_model.utils.lib_helpers as lib_commons
    #from tactus_model.utils.lib_tracker import Tracker # TODO  

    from tactus_yolov7 import Yolov7, resize

    from tactus_data import retracker
    from deep_sort_realtime.deepsort_tracker import DeepSort
    

MODEL_WEIGHTS_PATH = Path("data", "raw", "model", "yolov7-w6-pose.pt")
print(MODEL_WEIGHTS_PATH)
model_yolov7 = Yolov7(MODEL_WEIGHTS_PATH, device="cuda:0")   

#from deep_sort_realtime.deepsort_tracker import DeepSort
#from tactus_data import retracker



def par(path):  
    return ROOT + path if (path and path[0] != "/") else path


# Function for reading command line inputs

def get_command_line_arguments():

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Test action recognition on \n"
            "(1) a video, (2) a folder of images, (3) or web camera.")
        parser.add_argument("-m", "--model_path", required=False,
                            default='model/trained_classifier.pickle')
        parser.add_argument("-t", "--data_type", required=False, default='webcam',
                            choices=["video", "folder", "webcam"])
        parser.add_argument("-p", "--data_path", required=False, default="",
                            help="path to a video file, or images folder, or webcam. \n"
                            "For video and folder, the path should be "
                            "absolute or relative to this project's root. "
                            "For webcam, either input an index or device name. ")
        parser.add_argument("-o", "--output_folder", required=False, default='output/',
                            help="Which folder to save result to.")

        args = parser.parse_args()
        return args
    args = parse_args()
    if args.data_type != "webcam" and args.data_path and args.data_path[0] != "/":
        # If the path is not absolute, then its relative to the ROOT.
        args.data_path = ROOT + args.data_path
    return args


def get_dst_folder_name(src_data_type, src_data_path):
    ''' 
    This function sets the output folder name based on data_type and data_path
    Output format:
            DST_FOLDER/folder_name/vidoe.avi
            DST_FOLDER/folder_name/skeletons/XXXXX.txt
    '''

    assert(src_data_type in ["video", "folder", "webcam"])

    if src_data_type == "video":  
        folder_name = os.path.basename(src_data_path).split(".")[-2]

    elif src_data_type == "folder":  
        folder_name = src_data_path.rstrip("/").split("/")[-1]

    elif src_data_type == "webcam":
        # format : month-day-hour-minute-seconds
        folder_name = lib_commons.get_time_string()

    return folder_name


args = get_command_line_arguments()

SRC_DATA_TYPE = args.data_type
SRC_DATA_PATH = args.data_path
SRC_MODEL_PATH = args.model_path

DST_FOLDER_NAME = get_dst_folder_name(SRC_DATA_TYPE, SRC_DATA_PATH)

# Congiguration settings

cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["test.py"]

CLASSES = np.array(cfg_all["classes"])
SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

# Action recognition: number of frames used to extract features.
WINDOW_SIZE = int(cfg_all["features"]["window_size"])

# Output folder
DST_FOLDER = args.output_folder + "/" + DST_FOLDER_NAME + "/"
DST_SKELETON_FOLDER_NAME = cfg["output"]["skeleton_folder_name"]
DST_VIDEO_NAME = cfg["output"]["video_name"]
# framerate of output video.avi
DST_VIDEO_FPS = float(cfg["output"]["video_fps"])


# Video setttings

# If data_type is webcam, set the max frame rate.
SRC_WEBCAM_MAX_FPS = float(cfg["settings"]["source"]
                           ["webcam_max_framerate"])

# If data_type is video, set the sampling interval.
# TODO: check with the sampling interval: if set to 2 then the video reading is 2x faster
SRC_VIDEO_SAMPLE_INTERVAL = int(cfg["settings"]["source"]
                                ["video_sample_interval"])

# Openpose settings
#OPENPOSE_MODEL = cfg["settings"]["openpose"]["model"]
#OPENPOSE_IMG_SIZE = cfg["settings"]["openpose"]["img_size"]

# Display settings
img_disp_desired_rows = int(cfg["settings"]["display"]["desired_rows"])



def select_images_loader(src_data_type, src_data_path):
    if src_data_type == "video":
        images_loader = lib_images_io.ReadFromVideo(
            src_data_path,
            sample_interval=SRC_VIDEO_SAMPLE_INTERVAL)

    elif src_data_type == "folder":
        images_loader = lib_images_io.ReadFromFolder(
            folder_path=src_data_path)

    elif src_data_type == "webcam":
        if src_data_path == "":
            webcam_idx = 0
        elif src_data_path.isdigit():
            webcam_idx = int(src_data_path)
        else:
            webcam_idx = src_data_path
        images_loader = lib_images_io.ReadFromWebcam(
            SRC_WEBCAM_MAX_FPS, webcam_idx)
    return images_loader


class MultiPersonClassifier(object):
    ''' Function for recognizing actions of multiple people in a frame
    '''

    def __init__(self, model_path, classes):

        self.dict_id2clf = {}  # human id -> classifier of this person

        # Define a function for creating classifier for new people.
        self._create_classifier = lambda human_id: ClassifierOnlineTest(
            model_path, classes, WINDOW_SIZE, human_id)

    def classify(self, dict_id2skeleton):
        ''' Classify the action type of each skeleton in dict_id2skeleton '''

        # Clear people not in view
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():

            if id not in self.dict_id2clf:  # add this new person
                self.dict_id2clf[id] = self._create_classifier(id)

            classifier = self.dict_id2clf[id]
            print("skeleton:\n", skeleton)
            id2label[id] = classifier.predict(skeleton)  
            # predict label
            # print("\n\nPredicting label for human{}".format(id))
            # print("  skeleton: {}".format(skeleton))
            # print("  label: {}".format(id2label[id]))

        return id2label

    def get_classifier(self, id):
        ''' Get the classifier based on the person id.
        Arguments:
            id {int or "min"}
        '''
        if len(self.dict_id2clf) == 0:
            return None
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]


def remove_skeletons_with_few_joints(skeletons):
    ''' Remove bad skeletons before sending to the tracker '''
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[0:2+13*2:2]
        py = skeleton[1:2+13*2:2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)

        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
            # add this skeleton only when all requirements are satisfied
            good_skeletons.append(skeleton)
    return good_skeletons


def draw_result_img(img_disp, ith_img, humans, dict_id2skeleton,
                    skeleton_detector, multiperson_classifier):
    ''' Draw skeletons, labels, and prediction scores onto image for display '''


    # Draw all people's skeleton
    skeleton_detector.draw(img_disp, humans)

    # Draw bounding box and label of each person
    if len(dict_id2skeleton):
        for id, label in dict_id2label.items():
            skeleton = dict_id2skeleton[id]
            # scale the y data back to original
            skeleton[1::2] = skeleton[1::2] / scale_h
            # print("Drawing skeleton: ", dict_id2skeleton[id], "with label:", label, ".")
            lib_plot.draw_action_result(img_disp, id, skeleton, label, ith_img)

   
    cv2.putText(img_disp, "Frame:" + str(ith_img),
                (20, 20), fontScale=1.5, fontFace=cv2.FONT_HERSHEY_PLAIN,
                color=(0, 0, 0), thickness=2)


    if len(dict_id2skeleton):
        classifier_of_a_person = multiperson_classifier.get_classifier(
            id='min')

    return img_disp


def get_the_skeleton_data_to_save_to_disk(dict_id2skeleton):
    '''
    Save folloeing info per skeleton
        id, label, and the skeleton pionts of length 17*2.
    So the total length per row is 2+34=36
    '''
    skels_to_save = []
    for human_id in dict_id2skeleton.keys():
        label = dict_id2label[human_id]
        skeleton = dict_id2skeleton[human_id]
        skels_to_save.append([[human_id, label] + skeleton])#skeleton.tolist()])
    return skels_to_save


# Main function
if __name__ == "__main__":

    #Detector, tracker, classifier
 
    deepsort_tracker = DeepSort(n_init=3, max_age=5)
    '''
    Isssue: Tracking ID is False
    '''
    multiperson_classifier = MultiPersonClassifier(SRC_MODEL_PATH, CLASSES)

    # -- Image reader
    images_loader = select_images_loader(SRC_DATA_TYPE, SRC_DATA_PATH)

    # output folder
    os.makedirs(DST_FOLDER, exist_ok=True)
    os.makedirs(DST_FOLDER + DST_SKELETON_FOLDER_NAME, exist_ok=True)

    # video writer: TODO: write a new video write
    #video_writer = lib_images_io.VideoWriter(DST_FOLDER + DST_VIDEO_NAME, DST_VIDEO_FPS)

    # Read images and process
    try:
        ith_img = -1
        while images_loader.has_image():

            # -- Read image
            img = images_loader.read_image()
            
            ith_img += 1
            img_disp = img.copy()
            print(f"\nProcessing {ith_img}th image ...")

            # Skeleton detection
            
            img = resize(img)
            skeletons_  = model_yolov7.predict_frame(img)
            print(skeletons_)
            tracking_id = retracker.deepsort_track_frame(deepsort_tracker, img, skeletons_)
            print(tracking_id)
            if ith_img>2:
        
              skeleton_points=[]
              for human in skeletons_:
                skeleton_points.append([human["keypoints"][i] for i in range(len(human["keypoints"])) if (i+1) % 3 != 0])
            
              ske_dict = [(key, value)for i, (key, value) in enumerate(zip(tracking_id, skeleton_points))]
            
              yolo_tracker_dict = dict(ske_dict)            
          
              if len(yolo_tracker_dict):
                  dict_id2label = multiperson_classifier.classify(
                      yolo_tracker_dict)

              # Display results
              # Write a display function

              # Print label of a person
              if len(yolo_tracker_dict):
                  min_id = min(yolo_tracker_dict.keys())
                  print("prediced label is :", dict_id2label[min_id])

              # -- Display image, and write to video.avi
              #video_writer.write(img_disp)

              # -- Get skeleton data and save to file
              skels_to_save = get_the_skeleton_data_to_save_to_disk(yolo_tracker_dict)
              lib_commons.save_listlist(
                DST_FOLDER + DST_SKELETON_FOLDER_NAME +
                SKELETON_FILENAME_FORMAT.format(ith_img),
                skels_to_save)
          
    finally:
        #video_writer.stop()
        print("END OF TACTUS ACTION RECOGNITION")
