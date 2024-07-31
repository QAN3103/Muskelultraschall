# Project Seminar Medical System
Project by Quynh Anh Nguyen, Lea Grün, Dilan Mohamdi, Heyi Wang

## Short Overview
This project contains the necessary code to train an artificial neuronal network (ANN), a support vector machine model (SVM) and a logistic regression model to detect Muscle Atrophy from B-Mode Ultrasound Images. 

## Road Map
The code consists of the following main steps:

### 1) Manual Image Segmentation
#### 1. Manual Image Segmentation into Bone and Muscle Segments
Before feature engineering, a region of interest (ROI) from the bone and muscle was segmented from each image. For the bone segments, an image size of 256x256 pixels was set. In contrast, the muscle segments were chosen to be smaller, at 128x128 pixels, so that only the relevant muscle structures are included in the image segment. The bone and muscle segments are saved in [...]

### 2) Splitting the data into Training Set and Test Set and Calculating Features
The 'Features_Calculation_final.py' script demonstrates the feature calculation process. It begins by loading the reference data, followed by splitting the data into Training and Test set. For both sets, the script calculates features based on the segmented bone and muscle regions. Once the feature computation is complete, the results are saved into separate CSV files.

### 3) Initiate and train the ANN/SVM/Logistic Regression model
#### 1.
#### 2.
#### 3.

## Results
Each model is evaluated on the following five metrics: F1-Score, Accuracy, Precision, Recall, True Negative Rate (TNR). The results after training the ANN, SVM and LoRe are shown below
ANN: {F1: 0.9, Accuracy: 0.9, Precision: 0.96, Recall: 0.85, TNR: 0.95}
SVM: {F1: 0.88, Accuracy: 0.88, Precision: 0.88, Recall: 0.88, TNR: 0.88}
LoRe: {F1: 0.82, Accuracy: 0.83, Precision: 0.84, Recall: 0.81, TNR: 0.85}

After performing Feature Importance Analysis on LoRe, the top 13 features are: bone_Std_Hist, bone_Kurtosis_Hist, bone_Contrast, bone_Homegeneity, bone_Energy, bone_Correlation, bone_LRE, bone_GLNU, bone_RLNU, muscle_Mean_Hist, muscle_Std_Hist, muscle_Dissimilarity, muscle_Homogeneity. Note that this list does not imply a specific ranking or order.

## Resources and Acknowledgemens
The data for training is provided by Dr. Ilia Aroyo from Clinic for Neurology and Neurointensive Care Medicine in Darmstadt. This project also utilizes source code from Muhammad Razif Rizqullah, available at https://github.com/rizquuula. We would like to thank both for their valuable contributions to this project.
