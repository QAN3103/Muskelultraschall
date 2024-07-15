# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
#Install all the packages
# !pip install scikit-learn xgboost joblib seaborn pandas matplotlib
# !pip install opencv-python
# !pip install seaborn

# Install the required library
os.system('apt-get update')
os.system('apt-get install -y libgl1-mesa-glx')
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score
import joblib
from sklearn.metrics import confusion_matrix
import os
import cv2
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, f1_score, roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

#import train feature table. Path must be changed before use
reference = pd.read_csv('/work/reduced_features_train.csv')
#remore Image_ID columns
reference.drop('Image_ID', axis='columns', inplace=True)
reference

#create training data
X_train = reference.iloc [:, 1:]
y_train = reference.iloc [:, 0]

#initiate SVM model
model = SVC()

#feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)

# # 7# Define SVM parameter. Perform Grid Search with some-times(it coule be changed at cv=) Cross Validation
param_grid = {
'C': [0.01, 0.1, 1, 10, 100, 1000],
'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
'gamma': ['scale', 'auto']
}

# # Define F1 as scoring metrics
# #alternative: try accuray
f1_scorer = make_scorer(f1_score)

# # Initialise Hyperparameter Tuning
grid_search = GridSearchCV(model, param_grid, scoring=f1_scorer, cv=20, verbose = 1, n_jobs = -1)
grid_search.fit (X_train, y_train)

# #find out best parameter
final_param = grid_search.best_params_
final_score = grid_search.best_score_
print (f"Final parameter: {final_param}")
print (f"Final score: {final_score}")

# #find out best model
final_model = grid_search.best_estimator_

# #save model 
joblib.dump (final_model, 'SVM_model.pkl')

#Evaluation
#load feature matrix of test set
feature_test = pd.read_csv('/work/reduced_features_test.csv')
image_ids = feature_test['Image_ID']  # Save Image_IDs for later use
feature_test.drop('Image_ID', axis='columns', inplace=True)

X_test = feature_test.iloc [:, 1:]
y_test = feature_test.iloc [:, 0]

#feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns)

#create prediction
loaded_model = joblib.load('SVM_model.pkl')
y_pred = loaded_model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Identify false positives and false negatives
false_positives = image_ids[(y_test == 0) & (y_pred == 1)]
false_negatives = image_ids[(y_test == 1) & (y_pred == 0)]

# Print the results
print(f"False Positives (Image IDs): {false_positives.tolist()}")
print(f"False Negatives (Image IDs): {false_negatives.tolist()}")

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, y_pred)  # Use the positive class probabilities
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score (y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

# Identify false positives and false negatives
false_positives = image_ids[(y_test == 0) & (y_pred == 1)]
false_negatives = image_ids[(y_test == 1) & (y_pred == 0)]

# Print the results
print(f"False Positives (Image IDs): {false_positives.tolist()}")
print(f"False Negatives (Image IDs): {false_negatives.tolist()}")

# Function to find and load an image
def find_image(image_name, root_folder, image_type):
    """
    Find and load a grayscale image from a root folder.

    Parameters:
    image_name (str): The base name of the image file (e.g., 'TU1R_1').
    root_folder (str): The root directory to search for the image file.
    image_type (str): The file extension of the image (e.g., '.png' or '.jpg').

    Returns:
    image (numpy.ndarray): The loaded grayscale image.
    """
    for subfolder in ['_bone', '_muscle']:
        for root, _, files in os.walk(root_folder):
            for file in files:
                if file == image_name + subfolder + image_type:
                    absolute_path = os.path.join(root, file)
                    image = cv2.imread(absolute_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"Error: Could not load image from {absolute_path}. Please check the file path and file format.")
                    return image, subfolder
    print(f"Error: File not found for {image_name} in {root_folder}")
    return None, None

# Function to plot images
def plot_images(image_ids, root_folder, image_type, title):
    plt.figure(figsize=(10, 10))
    for i, image_id in enumerate(image_ids):
        image, subfolder = find_image(image_id, root_folder, image_type)
        if image is not None:
            plt.subplot(5, 5, i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"{image_id} ({subfolder})")
            plt.axis('off')
        if i >= 24:  # Show only up to 25 images
            break
    plt.suptitle(title)
    plt.show()

# Plot false positives
plot_images(false_positives, '/work/Train', '.jpg', 'False Positives')

# Plot false negatives
plot_images(false_negatives, '/work/Train', '.jpg', 'False Negatives')
