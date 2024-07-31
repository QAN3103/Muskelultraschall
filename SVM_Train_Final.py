"""
Script for SVM Model Training
This script is intended for training a Support Vector Machine (SVM) model for ultrasonic-image classification tasks.
The process involves data loading, features scaling, model training, print best parameters and the save the best model.
It utilizes several Python libraries including scikit-learn, joblib and pandas.

Authors:
- Quỳnh Anh Nguyễn
- Heyi Wang
- Dilan Mohammadi
- Lea Grün

Functions: 
main(): the whole workflow of this script, including load train dataset, scaling features, train and find the best model, finally it save the
best model in "SVM_model.pkl".

Requirements:
Python 3.x
Libraries: scikit-learn, joblib, pandas
"""

#Import all the libiaries
"""
Import necessary libraries for data manipulation and machine learning
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

def main():
    # Import train feature table. Path must be changed before use
    reference = pd.read_csv('features_train.csv')
    # Remore Image_ID columns
    reference.drop('Image_ID', axis='columns', inplace=True)
    reference
    
    # Create training data
    X_train = reference.iloc [:, 1:]
    y_train = reference.iloc [:, 0]
    
    # Initiate SVM model
    model = SVC()
    
    # Feature scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    
    # Define SVM parameter. Perform Grid Search with some-times(it coule be changed at cv=) Cross Validation
    param_grid = {
    'C': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
    }
    
    ## Define F1 as scoring metrics
    f1_scorer = make_scorer(f1_score)
    
    # Initialise Hyperparameter Tuning
    grid_search = GridSearchCV(model, param_grid, scoring=f1_scorer, cv=20, verbose = 1, n_jobs = -1)
    grid_search.fit (X_train, y_train)
    
    # Find out best parameter
    final_param = grid_search.best_params_
    final_score = grid_search.best_score_
    print (f"Final parameter: {final_param}")
    
    # Find out best model
    final_model = grid_search.best_estimator_
    
    # Save the best model 
    joblib.dump (final_model, 'SVM_model.pkl')
    

# call main in jupyternotebook
main()




