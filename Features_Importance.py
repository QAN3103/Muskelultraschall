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

# # Features Importance Research

# ## 1. PERMUTATION FEATURES IMPORTANCE

# +
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler

# Load training feature table. Path must be changed before use
reference = pd.read_csv('/work/comparision/features_train.csv')

# Remove Image_ID columns
reference.drop('Image_ID', axis='columns', inplace=True)

# Create training data
X_train = reference.iloc[:, 1:]
y_train = reference.iloc[:, 0]

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# Load feature matrix of test set
feature_test = pd.read_csv('/work/comparision/features_test.csv')
feature_test.drop('Image_ID', axis='columns', inplace=True)

# Align the columns of X_test to match X_train
X_test = feature_test[X_train.columns]

# Feature scaling
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
y_test = feature_test.iloc[:, 0]

# Load the model
loaded_model = joblib.load('SVM_model11.pkl')

# Compute permutation feature importance
r = permutation_importance(loaded_model, X_test, y_test, n_repeats=30, random_state=42)

# Get feature names from training data
feature_names = X_train.columns

# Print the feature importances
for i in r.importances_mean.argsort()[::-1]:
    print(f"{feature_names[i]:<20} {r.importances_mean[i]:.3f} +/- {r.importances_std[i]:.3f}")
# -

# ## SHAP Shapely additive Explanations

# ### 2.1 Kernel Explainer

# +
# !pip install shap
import shap
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Load training feature table. Path must be changed before use
reference = pd.read_csv('/work/comparision/features_train.csv')

# Remove Image_ID column
reference.drop('Image_ID', axis='columns', inplace=True)

# Create training data
X_train = reference.iloc[:, 1:]
y_train = reference.iloc[:, 0]

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# Load feature matrix of test set
feature_test = pd.read_csv('/work/comparision/features_test.csv')
feature_test.drop('Image_ID', axis='columns', inplace=True)

# Align the columns of X_test to match X_train
X_test = feature_test[X_train.columns]

# Feature scaling
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
y_test = feature_test.iloc[:, 0]

# Initialize SVM with polynomial kernel
svm_poly = make_pipeline(StandardScaler(), SVC(kernel='poly', degree=3))

# Fit model
svm_poly.fit(X_train, y_train)

# Load the model
loaded_model = joblib.load('SVM_model11.pkl')

# KernelExplainer
explainer = shap.KernelExplainer(loaded_model.predict, X_train)
shap_values = explainer.shap_values(X_test, nsamples=100)

# Get feature names from training data
feature_names = X_train.columns

# Plot the SHAP values for each feature
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Plot the SHAP values as a bar chart
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names)


# -

# ### 2.2 Permutation Explainer

# +
# !pip install shap
import shap
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Load training feature table. Path must be changed before use
reference = pd.read_csv('/work/comparision/features_train.csv')

# Remove Image_ID column
reference.drop('Image_ID', axis='columns', inplace=True)

# Create training data
X_train = reference.iloc[:, 1:]
y_train = reference.iloc[:, 0]

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# Load feature matrix of test set
feature_test = pd.read_csv('/work/comparision/features_test.csv')
feature_test.drop('Image_ID', axis='columns', inplace=True)

# Align the columns of X_test to match X_train
X_test = feature_test[X_train.columns]

# Feature scaling
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
y_test = feature_test.iloc[:, 0]

# Initialize SVM with polynomial kernel
svm_poly = make_pipeline(StandardScaler(), SVC(kernel='poly', degree=3))

# Fit model
svm_poly.fit(X_train, y_train)

# Load the model
loaded_model = joblib.load('SVM_model11.pkl')

# KernelExplainer
explainer = shap.PermutationExplainer(loaded_model.predict, X_train)
shap_values = explainer.shap_values(X_test)

'''
PermutationExplainer:
shap.PermutationExplainer is also model-agnostic and calculates SHAP values by permuting the feature values and 
measuring the impact on the model's output.
It is typically faster than KernelExplainer but may be less accurate 
because it relies on the assumption that the feature contributions are independent.
'''

# Get feature names from training data
feature_names = X_train.columns

# Plot the SHAP values for each feature
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Plot the SHAP values as a bar chart
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names)


