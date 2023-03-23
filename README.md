<br />
<div align="center">
  <h1>TACTUS - model pipeline</h1>
  <h3>Threatening activities classification toward users' security</h3>
</div>


## 1. Overview
This repository contains the implementation for action classifier model of the TACTUS project. Along with classifier Training, Testing, and Output generation, this repo has  following functionalities
#### Combining all the keypoint features from pose detector in a structured manner:
Input format:TBD

Output format:TBD

#### Preprocess the input data and extract features: TBD

#### Offline Training:TBD

#### Online Testing:TBD

## 2. Structure of the TACTUS Classifier TBD

## 3. Source Scripts TBD

## 4. How to run
### Step 1: Installling dependencies
install the dependencies in file `requirements.txt`

### Step 2: Combine all keypoints from pose detector and tracker
```
!python src/combine_skeletons_info.py
```

This creates an json formatted file in the following path:
```
data_proc/raw_skeletons/skeletons_data.txt
```

### Run pre-processing to extract the useful features
```
!python src/preprocessing_features.py
```
This script creates an input features file in the .csv format for classifier

``` data_proc/features_X.csv ``` for features and 
``` data_proc/features_Y.csv ``` for action labels


### Training the model
```
!python src/train.py
```
### Testing the TACTUS classifier Model
This script is in progress right now ...
```
!python src/test.py
```



# Useful ressources
- [Write a better commit message](https://gist.github.com/MarcBresson/dd57a17f2ae60b6cb8688ee64cd7671d)
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
