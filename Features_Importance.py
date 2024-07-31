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

# + [markdown] formattedRanges=[] cell_id="c90e5cde99d542b38f5dc99c693a9f29" deepnote_cell_type="text-cell-h2"
# ## Features Importance Research

# + [markdown] formattedRanges=[] cell_id="36ea775cdcf0466194554f069286882e" deepnote_cell_type="text-cell-h3"
# ### 1. PERMUTATION FEATURES IMPORTANCE

# + source_hash execution_start=1720014476682 execution_millis=1941 deepnote_to_be_reexecuted=false cell_id="7b904470732548ff905029b79bdca43e" deepnote_cell_type="code"
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

# +
### 2.SHAP (SHapley Additive exPlanations)###

###2.1 KernelExplainer###

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

'''
Key Differences
Type of Explainer:
KernelExplainer:
shap.KernelExplainer is a model-agnostic method that approximates SHAP values by treating the model as a black box. 
It uses a sampling approach to estimate SHAP values and can be applied to any machine learning model.
It is generally more accurate but computationally expensive, especially with complex models and large datasets.
'''

# Get feature names from training data
feature_names = X_train.columns

# Plot the SHAP values for each feature
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Plot the SHAP values as a bar chart
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names)

# + source_hash execution_start=1720014346950 execution_millis=10692 deepnote_to_be_reexecuted=false cell_id="c9e0a37e86c041569c62ed0f1e9d8dca" deepnote_cell_type="code"
### 2.2 PermutationExplainer ###

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



# + [markdown] formattedRanges=[] cell_id="a9daf0d37cd94119a7d642b48ebac4d7" deepnote_cell_type="text-cell-h2"
# ## Try to find the most important features

# + [markdown] formattedRanges=[] cell_id="5340f3c60026472ea4217e46ded6ac0c" deepnote_cell_type="text-cell-h3"
# ### Using RFE

# + cell_id="3295eb60dec14d89beb1e7f38e2ae32c" deepnote_cell_type="code"
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import pandas as pd

# load the features csv
reference = pd.read_csv('/work/comparision/features_train.csv')
reference.drop('Image_ID', axis='columns', inplace=True)
X_train = reference.iloc[:, 1:]
y_train = reference.iloc[:, 0]

# Load SVM Model
model = SVC(kernel='linear')

# use REF to select features
selector = RFE(model, n_features_to_select=5, step=1)
selector = selector.fit(X_train, y_train)

# Print the selected features
selected_features = X_train.columns[selector.support_]
print("Selected Features:", selected_features)


# + [markdown] formattedRanges=[] cell_id="259203ba8e724d788fd85dba234071b2" deepnote_cell_type="text-cell-h3"
# ### Using Lasso

# + cell_id="650374ac9c8949b48fbf4672142ee6db" deepnote_cell_type="code"
#from sklearn.linear_model import LassoCV
import pandas as pd

# Load the features csv
reference = pd.read_csv('/work/comparision/features_train.csv')
reference.drop('Image_ID', axis='columns', inplace=True)
X_train = reference.iloc[:, 1:]
y_train = reference.iloc[:, 0]

# Use LassoCV to select features
lasso = LassoCV(cv=5)
lasso.fit(X_train, y_train)

# Calculate the coefficients for features
lasso_coef = pd.Series(lasso.coef_, index=X_train.columns)

# Print Features with non-zero coefficients
selected_features = lasso_coef[lasso_coef != 0].index
print("Selected Features:", selected_features)


# + [markdown] formattedRanges=[] cell_id="a3e7cd3fcd3846689b67a5f13aaccc5d" deepnote_cell_type="text-cell-h3"
# ### Using XGBoost

# + cell_id="d8bb93998b244d3ab456a4c2aa824661" deepnote_cell_type="code"
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the features csv
reference = pd.read_csv('/work/comparision/features_train.csv')
reference.drop('Image_ID', axis='columns', inplace=True)
X_train = reference.iloc[:, 1:]
y_train = reference.iloc[:, 0]

# Ttrain XGBoost model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Calculate feature importance
feature_importance = xgb_model.feature_importances_
features = X_train.columns

# Create a dataframe with feature importance and feature names
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
})

# Select top N features
N = 5  # Number of top features to select, it's up to you
top_features = importance_df.sort_values(by='Importance', ascending=False).head(N)
print(f"Top {N} Features:", top_features['Feature'].values)

# Plot feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title(f'Top {N} Feature Importance')
plt.show()
