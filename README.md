# Project Seminar Medical System
Project by Quynh Anh Nguyen, Lea Grün, Dilan Mohamdi, Heyi Wang

## Short Overview
This project contains the necessary code to train an artificial neural network (ANN), a support vector machine model (SVM), and a logistic regression (LoRe) model to detect Muscle Atrophy from B-Mode ultrasound images. The dataset consists of 258 images, including 129 normal images and 129 images showing pathological changes. Each image is assigned a hypothetical Patient_ID in the format 'TU_xx_123'. Although the images are classified into four grades according to Heckmatt and Dubowitz [1], this project distinguishes only between normal findings (Grade 1) and pathological findings (Grades 2 to 4).

## Road Map
The project consists of the following main steps:

### 1) Manual Image Segmentation into Bone and Muscle Segments
Before feature engineering, a region of interest (ROI) from the bone and muscle was manually segmented from each image. For the bone segments, an image size of 256x256 pixels was set. In contrast, the muscle segments were chosen to be smaller, at 128x128 pixels, to ensure that only the relevant muscle structures were included in the image segment. The dataset including their corresponding grades and the segmented images can be found at: https://hessenbox.tu-darmstadt.de/getlink/fiVDHQ1W6ENZpitGHe3EPTJ7/

### 2) Splitting the data into Training Set and Test Set and Calculating Features
The 'Features_Calculation_final.py' script demonstrates the feature calculation process. It begins by loading the reference data and then splits it into training and test sets. A validation set is not used due to the limited data available. For both sets, the script calculates features based on the segmented bone and muscle regions. Once the feature computation is complete, the results are saved into separate CSV files. 
#### read_image_ids(csv_path): 
Reads image IDs from a CSV file.
#### find_image(image_name, root_folder, image_type, suffix): 
Finds and loads a grayscale image from a root folder with a given suffix.
#### create_hist(image): 
Creates a histogram for a grayscale image.
#### calculate_hist(hist): 
Calculates statistical properties of a histogram: Mean, Median, Standard Deviation, Skewness, Kurtosis
#### calculate_glcm(image, distances, angles, levels): 
Calculates the GLCM features: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, Entropy
#### calculate_glrlm(image): 
Calculates GLRLM features: SRE (Short Runs Emphasis), LRE (Long Runs Emphasis), GLNU (Gray Level Nonuniformity), RLNU (Run Length Nonuniformity) and RP (Run Percentage)
#### process_images(image_ids, root_folder, output_csv, original_df): 
Calculates features from bone and muscle segments of the Training set and Test set, and save results to a new CSV.

### 3) Initiate and train the ANN/SVM/LoRe model
Before Training, all features must be normalised between (0,1). Cross-validation is incorporated into the training process to compensate for the missing validation set and to ensure robustness. In this project, the following three models are implemented:

#### 1. ANN Model
The model architecture is constructed using the Keras API. The model begins with an input layer of 32 neurons, followed by a hidden layer of 16 neurons. The output layer consists of a single unit utilizing a sigmoid activation function to classify between 'sick' and 'healthy'. In addition, a combination of L1 and L2 regularization and batch normalization are used to improve generalization and model stability. Hyperparameter-Tuning is also performed using GridSearchCV to identify the optimal batch size and the number of epochs.

After Hyperparameter-Tuning, the ANN model was trained with an optimal batch size of 8 and an optimal epoch count of 50. If the validation loss does not improve after 10 epochs, the learning rate will be reduced by a factor of 0.1, with a lower boundary of 0.0000001. Early stopping is also implemented when the validation loss doesn't improve for 20 epochs. In the end, the best model is saved as 'ANN_model.keras'.

#### 2. SVM Model
In an SVM model, the primary parameters influencing performance are C, kernel type (linear, rbf, poly, sigmoid), and gamma setting (scale, auto). To identify the optimal combination of these parameters, Hyperparameter-Tuning is performed using GridSearchCV, with the F1 score as the evaluation metric.

After hyperparameter tuning, the best parameter combination was found to be {C = 0.6, kernel = 'poly', and gamma = 'scale'}. The optimal model is then saved as 'SVM_model.pkl'.

#### 3. LoRe Model
In a LoRe model, the primary parameters influencing performance are C, solver (liblinear, lbfgs, newton-cg, sag, saga), and penalty (l1, l2, None). To determine the optimal combination of these parameters, Hyperparameter-Tuning is conducted using GridSearchCV with the F1 score as the evaluation metric.

After hyperparameter tuning, the best parameter combination was found to be {C = 0.001, solver = 'saga', and penalty = None}. The optimal model is then saved as 'LoRe_model.pkl'.

To further refine the LoRe model, a Feature Importance Analysis using RFECV is conducted. This involves systematically removing features (Feature Ablation) and retraining the model. The goal is to identify the feature combination that achieves the highest F1 score through cross-validation.

### 4) Create prediction
Predictions on the Test set can be made by calling model.predict(). The outputs of the SVM and LoRe model are binary classifications, with '1' indicating 'Sick' and '0' indicating 'Healthy.' The output of the ANN spans a continuous range between 0 and 1. To convert this continuous output into a binary classification, a threshold is applied. If the output exceeds the threshold, it is classified as '1' (Sick); otherwise, it is classified as '0' (Healthy). In this project, the optimal threshold was determined to be 0.4.

## Results
Each model is evaluated on the following five metrics: F1-Score, Accuracy, Precision, Recall, True Negative Rate (TNR). The results after training the ANN, SVM and LoRe are shown below:

ANN: {F1: 0.9, Accuracy: 0.9, Precision: 0.96, Recall: 0.85, TNR: 0.95}

SVM: {F1: 0.88, Accuracy: 0.88, Precision: 0.88, Recall: 0.88, TNR: 0.88}

LoRe: {F1: 0.82, Accuracy: 0.83, Precision: 0.84, Recall: 0.81, TNR: 0.85}

After performing Feature Importance Analysis on LoRe, the top 13 features are: {bone_Std_Hist, bone_Kurtosis_Hist, bone_Contrast, bone_Homegeneity, bone_Energy, bone_Correlation, bone_LRE, bone_GLNU, bone_RLNU, muscle_Mean_Hist, muscle_Std_Hist, muscle_Dissimilarity, muscle_Homogeneity}. Note that this list does not imply a specific ranking or order.

## Resources and Acknowledgemens
The data for training is provided by Dr. Ilia Aroyo from Clinic for Neurology and Neurointensive Care Medicine in Darmstadt. This project also utilizes source code from Muhammad Razif Rizqullah, available at https://github.com/rizquuula. We would like to thank both for their valuable contributions to this project.

[1] Heckmatt, J. Z., Leeman, S., & Dubowitz, V. (1982). Ultrasound imaging in the diagnosis of muscle disease. The Journal of pediatrics, 101(5), 656–660. https://doi.org/10.1016/s0022-3476(82)80286-2

## Possible Debugging
### In case of UnicodeEncodeError, please run the following code: 

import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

### To handle TensorFlow Warnings, please run the following code:

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


