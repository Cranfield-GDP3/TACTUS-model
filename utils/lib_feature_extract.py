''' This script defines functions/class to process features:

* def extract_multi_frame_features
    Convert raw skeleton data into features extracted from multiple frames
    by calling `class FeatureGenerator`.

* class FeatureGenerator:
    Compute features from a video sequence of raw skeleton data.

'''


import numpy as np
import math
from collections import deque

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    from tools.an_example_skeleton_of_standing import get_a_normalized_standing_skeleton

# -- Settings
NOISE_INTENSITY = 0.05

# -- Constant

PI = np.pi
Inf = float("inf")
NaN = 0


def retrain_only_body_joints(skeleton):
    ''' 
    Redundant Joints for this usecase
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    are removed 

    Neck point or center point is calculated using the midppoint of 3: LEar and 4: REar
    So, we habe total 26 coordinates now after removal of the first 5 keypoints and 
    addition of keypoint NECK
    '''
    new_skeleton = skeleton.copy()[10:]
    
    neck_x, neck_y = (skeleton.copy()[6] + skeleton.copy()[8])/2, (skeleton.copy()[7] + skeleton.copy()[9])/2
    neck_ = np.array([neck_x, neck_y])
  
    retrained_skeleton = np.concatenate((neck_,new_skeleton), axis=0)

    return retrained_skeleton
   

# Keypoints format

TOTAL_JOINTS = 13
NECK = 0
L_ARMS = [1, 3, 5]
R_ARMS = [2, 4, 6]
L_KNEE = 9
L_ANKLE = 11
R_KNEE = 10
R_ANKLE = 12
L_LEGS = [9, 11] #knee and ankle
R_LEGS = [10, 12]
ARMS_LEGS = L_ARMS + R_ARMS + L_LEGS + R_LEGS
L_THIGH = 7
R_THIGH = 8

STAND_SKEL_NORMED = retrain_only_body_joints(
    get_a_normalized_standing_skeleton())

# -- Functions


def extract_multi_frame_features(
        X, Y, video_indices, window_size,
        is_adding_noise=False, is_print=False):
    ''' 
        Extract features of body velocity, joint velocity, and normalized joint positions.
    '''
    X_new = []
    Y_new = []
    N = len(video_indices)

    # Loop through all videos
    for i, _ in enumerate(video_indices):

        # If a new video starts : reset feature generator
        # TODO: remove print statements
        if i == 7:
          print(i, video_indices[i])

        if i == 0 or video_indices[i] != video_indices[i-1]:
            fg = FeatureGenerator(window_size, is_adding_noise)
        
        # Get features
        success, features = fg.add_cur_skeleton(X[i, :], video_indices[i])
        if success:  
            X_new.append(features)
            Y_new.append(Y[i])

        # Print
        if is_print and i % 1000 == 0:
            print(f"{i}/{N}", end=", ")
            
    if is_print:
        print("")
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
  
    return X_new, Y_new

# Helper functions for math operations
class Math():

    @staticmethod
    def calc_dist(p1, p0):
        return math.sqrt((p1[0]-p0[0])**2+(p1[1]-p0[1])**2)

    @staticmethod
    def pi2pi(x):
        if x > PI:
            x -= 2*PI
        if x <= -PI:
            x += 2*PI
        return x

    @staticmethod
    def calc_relative_angle(x1, y1, x0, y0, base_angle):
        # compute rotation from {base_angle} to {(x0,y0)->(x1,y1)}
        if (y1 == y0) and (x1 == x0):
            return 0
        a1 = np.arctan2(y1-y0, x1-x0)
        return Math.pi2pi(a1 - base_angle)

    @staticmethod
    def calc_relative_angle_v2(p1, p0, base_angle):
        # compute rotation from {base_angle} to {p0->p1}
        return Math.calc_relative_angle(p1[0], p1[1], p0[0], p0[1], base_angle)



# Helper functions for skeleton joints 

def get_joint(x, idx):
    px = x[2*idx]
    py = x[2*idx+1]
    return px, py


def set_joint(x, idx, px, py):
    x[2*idx] = px
    x[2*idx+1] = py
    return


def check_joint(x, idx):
    return x[2*idx] != NaN


class ProcFtr(object):

    @staticmethod
    def drop_arms_and_legs_randomly(x, thresh=0.3):
        ''' Randomly drop one arm or one leg with a probability of thresh '''
        x = x.copy()

        N = len(ARMS_LEGS)
        rand_num = np.random.random()
        if rand_num < thresh:
            joint_idx = int((rand_num / thresh)*N)
            set_joint(x, joint_idx, NaN, NaN)
        return x

    @staticmethod
    def has_neck_and_thigh(x):
        ''' Check if a skeleton has a neck and at least one thigh '''
        return check_joint(x, NECK) and (check_joint(x, L_THIGH) or check_joint(x, R_THIGH))

    @staticmethod
    def get_body_height(x):
        ''' Compute height of the body, which is defined as:
            the distance between `neck` and `thigh`.
        '''
        x0, y0 = get_joint(x, NECK)

        # Get average thigh height
        x11, y11 = get_joint(x, L_THIGH)
        x12, y12 = get_joint(x, R_THIGH)
        if y11 == NaN and y12 == NaN:  # Invalid data
            return 1.0
        if y11 == NaN:
            x1, y1 = x12, y12
        elif y12 == NaN:
            x1, y1 = x11, y11
        else:
            x1, y1 = (x11 + x12) / 2, (y11 + y12) / 2

        # Get body height
        height = ((x0-x1)**2 + (y0-y1)**2)**(0.5)
        return height

    @staticmethod
    def remove_body_offset(x):
        ''' The origin is the neck.
        TODO: Deal with empty data.
        '''
        x = x.copy()
        px0, py0 = get_joint(x, NECK)
        x[0::2] = x[0::2] - px0
        x[1::2] = x[1::2] - py0
       
        return x

    @staticmethod
    # TODO: computation is in progress, will be updated in next commit --------
    def joint_pos_2_angle_and_length(x):

        class JointPosExtractor(object):
            def __init__(self, x):
                self.x = x
                self.i = 0

            def get_next_point(self):
                p = [self.x[self.i], self.x[self.i+1]]
                self.i += 2
                return p
        joints_ = JointPosExtractor(x)

        pneck = joints_.get_next_point()
        plshoulder = joints_.get_next_point()
        prshoulder = joints_.get_next_point()
        plelbow = joints_.get_next_point()
        prelbow = joints_.get_next_point()
        plwrist = joints_.get_next_point()
        prwrist = joints_.get_next_point()
        plhip = joints_.get_next_point()
        prhip = joints_.get_next_point()
        plknee = joints_.get_next_point()
        prknee = joints_.get_next_point()
        plankle = joints_.get_next_point()
        prankle = joints_.get_next_point()

        class Get12Angles(object):
            def __init__(self):
                self.j = 0
                self.f_angles = np.zeros((12,))
                self.x_lengths = np.zeros((12,))

            def set_next_angle_len(self, next_joint, base_joint, base_angle):
                angle = Math.calc_relative_angle_v2(
                    next_joint, base_joint, base_angle)
                dist = Math.calc_dist(next_joint, base_joint)
                self.f_angles[self.j] = angle
                self.x_lengths[self.j] = dist
                self.j += 1

        angle_ = Get12Angles()

        angle_.set_next_angle_len(prshoulder, pneck, PI)  # r-shoulder
        angle_.set_next_angle_len(prelbow, prshoulder, PI/2)  # r-elbow
        angle_.set_next_angle_len(prwrist, prelbow, PI/2)  # r-wrist

        angle_.set_next_angle_len(plshoulder, pneck, 0)  
        angle_.set_next_angle_len(plelbow, plshoulder, PI/2)  
        angle_.set_next_angle_len(plwrist, plelbow, PI/2)  

        angle_.set_next_angle_len(prhip, pneck, PI/2+PI/18)
        angle_.set_next_angle_len(prknee, prhip, PI/2)
        angle_.set_next_angle_len(prankle, prknee, PI/2)

        angle_.set_next_angle_len(plhip, pneck, PI/2-PI/18)
        angle_.set_next_angle_len(plknee, plhip, PI/2)
        angle_.set_next_angle_len(plankle, plknee, PI/2)

        # Output
        features_angles = angle_.f_angles
        features_lens = angle_.x_lengths
        return features_angles, features_lens

# Featuregenerator is a main class for extracting the features from input data

class FeatureGenerator(object):
    def __init__(self,
                 window_size,
                 is_adding_noise=False):
        '''
        Arguments:
            window_size {int}: Number of adjacent frames 
            is_adding_noise {bool}: For data augmentation
            noise_intensity {float}: Adding noise relative to the skeleton height
        '''
        self._window_size = window_size
        self._is_adding_noise = is_adding_noise
        self._noise_intensity = NOISE_INTENSITY
        self.reset()

    def reset(self):
        ''' Reset the FeatureGenerator '''
        self._x_deque = deque()
        self._angles_deque = deque()
        self._lens_deque = deque()
        self._pre_x = None

    def add_cur_skeleton(self, skeleton, indices):
        ''' Input: 26 keypoints skeleton
            Output: Extracted features
        '''
        # TODO: remove print statements
        if indices ==1:
          print("Original values+++++++++++++++\n", skeleton)

        x = retrain_only_body_joints(skeleton)

        if indices ==1:
          print("Useful values+++++++++++++++\n", x)
        
        if not ProcFtr.has_neck_and_thigh(x):
            print("no neck thigh")
            self.reset()
            return False, None

        else:
            ''' The input skeleton has a neck and at least one thigh '''
            # Preprocess x      
            x = self._fill_invalid_data(x)
            if self._is_adding_noise:
                # Add noise druing training stage to augment data (TODO: remove in next commit)
                x = self._add_noises(x, self._noise_intensity)
            x = np.array(x)
            angles, lens = ProcFtr.joint_pos_2_angle_and_length(x)
            if indices ==1:
              print("orginal angles ----------\n", angles)
              print("orginal lens ----------\n", lens)
            

            # Push to deque
            self._x_deque.append(x)
            self._angles_deque.append(angles) 
            self._lens_deque.append(lens) 
            self._maintain_deque_size()
            self._pre_x = x.copy()

            # -- Extract features
            if len(self._x_deque) < self._window_size:
                return False, None
            else:
                # -- Feature Normalization ------ 
                h_list = [ProcFtr.get_body_height(xi) for xi in self._x_deque]
                mean_height = np.mean(h_list)
                xnorm_list = [ProcFtr.remove_body_offset(xi)/mean_height
                              for xi in self._x_deque]

                if indices ==1:
                  print("Normalized values++++++++++++++++++\n", xnorm_list,mean_height)

                # -- Get features: Right now only normalized keypoints are used
                #TODO: add angles
                f_poses = self._deque_features_to_1darray(xnorm_list)

                f_angles = self._deque_features_to_1darray(self._angles_deque) 
                f_lens = self._deque_features_to_1darray(self._lens_deque) / mean_height 
                
                if indices ==1:
                  print("f_angles-------------------\n", f_angles)
                  print("f_lens-------------------\n", f_lens)

                # -- Get features of motion
                f_v_center = self._compute_v_center(
                    self._x_deque, step=1) / mean_height  # len = (t=4)*2 = 8
                
                if indices == 1:
                  print(self._x_deque, mean_height)
                  print("Center of Mass Velocity: ++++++++++++++++++++++++++++:\n", f_v_center)

                f_v_center = np.repeat(f_v_center, 10) 

                f_v_joints = self._compute_v_all_joints(
                    xnorm_list, step=1)  # len = (t=(5-1)/step)*13*2 = 104
                if indices== 1:
                  print("Joints velocity: ++++++++++++++++++++++++++:\n", f_v_joints)


                # lengths :130 104 80 60 60 =  434
                features = np.concatenate((f_poses, f_v_joints, f_v_center, f_angles, f_lens))
                if indices ==1 :
                  print("lenghts *****\n", len(f_poses), len(f_v_joints), len(f_v_center), len(f_angles), len(f_lens))
                  print("Final Feature Vector: ++++++++++++++++++++++++\n", len(features),features)

                
                return True, features.copy()

    def _maintain_deque_size(self):
        if len(self._x_deque) > self._window_size:
            self._x_deque.popleft()
        if len(self._angles_deque) > self._window_size:
            self._angles_deque.popleft()
        if len(self._lens_deque) > self._window_size:
            self._lens_deque.popleft()

    def _compute_v_center(self, x_deque, step):
        vel = []
        for i in range(0, len(x_deque) - step, step):
            dxdy = x_deque[i+step][0:2] - x_deque[i][0:2]
            vel += dxdy.tolist()

        return np.array(vel)

    def _compute_v_all_joints(self, xnorm_list, step):
        vel = []
 
        for i in range(0, len(xnorm_list) - step, step):
            dxdy = xnorm_list[i+step][:] - xnorm_list[i][:]
            vel += dxdy.tolist()
        return np.array(vel)


    def _fill_invalid_data(self, x):
        ''' Fill the missing  elements in x with
            their relative-to-neck position in the preious previsious frame or in this case 'x'
        Argument:
            x : this {np.array} contains the skeleton points which has atleast a neck and hip keypoints
        '''
        res = x.copy()

        def get_px_py_px0_py0(x):
            px = x[0::2]  # list of x
            py = x[1::2]  # list of y
            px0, py0 = get_joint(x, NECK)  # neck
            return px, py, px0, py0
        cur_px, cur_py, cur_px0, cur_py0 = get_px_py_px0_py0(x)
        cur_height = ProcFtr.get_body_height(x)

        is_lack_knee = check_joint(x, L_KNEE) or check_joint(x, R_KNEE)
        is_lack_ankle = check_joint(x, L_ANKLE) or check_joint(x, R_ANKLE)
        if (self._pre_x is None) or is_lack_knee or is_lack_ankle:
            # If preious data is invalid or there is no knee or ankle,
            # then fill the data based on the STAND_SKEL_NORMED.
            for i in range(TOTAL_JOINTS*2):
                if res[i] == NaN:
                    res[i] = (cur_px0 if i % 2 == 0 else cur_py0) + \
                        cur_height * STAND_SKEL_NORMED[i]
            return res

        pre_px, pre_py, pre_px0, pre_py0 = get_px_py_px0_py0(self._pre_x)
        pre_height = ProcFtr.get_body_height(self._pre_x)

        scale = cur_height / pre_height

        bad_idxs = np.nonzero(cur_px == NaN)[0]
        if not len(bad_idxs):  # No invalid data
            return res

        cur_px[bad_idxs] = cur_px0 + (pre_px[bad_idxs] - pre_px0) * scale
        cur_py[bad_idxs] = cur_py0 + (pre_py[bad_idxs] - pre_py0) * scale
        res[::2] = cur_px
        res[1::2] = cur_py
        return res

    def _add_noises(self, x, intensity):
        ''' Add noise to x with a ratio relative to the body height '''
        height = ProcFtr.get_body_height(x)
        randoms = (np.random.random(x.shape, ) - 0.5) * 2 * intensity * height
        x = [(xi + randoms[i] if xi != 0 else xi)
             for i, xi in enumerate(x)]
        return x

    def _deque_features_to_1darray(self, deque_data):
        ''' convert the extracted features in a 1D array '''
        features = []
        for i in range(len(deque_data)):
            next_feature = deque_data[i].tolist()
            features += next_feature
        features = np.array(features)
        return features

    def _deque_features_to_2darray(self, deque_data):
        # TODO: for multiple skeleton action detection
        features = []
        for i in range(len(deque_data)):
            next_feature = deque_data[i].tolist()
            features.append(next_feature)
        features = np.array(features)
        return features
