"""
Muscle Atrophy Detection with Support Vector Machine (SVM)

This script is intended to develop and train a SVM model for the purpose of muscle atrophy detection. The process includes the following steps:
1. Importing the train feature table.
2. Separating feature matrix, defined as X_train, and label vector, defined as y_train.
3. Feature scaling using MinMaxScaler between (0,1). 
4. Performing GridSearch to find the best parameter combination of C - Kernel - Gamm. With each parameter combination, a model is trained using cross validation and evaluated based on F1-Score. 
5. Saving the best model and printing out the best parameter combination.

Authors:
- Quỳnh Anh Nguyễn
- Heyi Wang
- Dilan Mohammadi
- Lea Grün

Functions: 
main(): Initiate the training process.

Requirements:
Python 3.x
Libraries: scikit-learn, joblib, pandas, joblib
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

def main():
    """
    Main function to execute the training process.
    """
    # Import the training feature table. 
    # Note: The file path must be updated to the appropriate location before use.
    reference = pd.read_csv('C:/Users/Quynh Anh/Muskelultraschall/features_train.csv')
    
    # Remove the 'Image_ID' column as it is not needed for training
    reference.drop('Image_ID', axis='columns', inplace=True)
    
    # Create training data by separating features and labels.
    # X_train will contain all columns except the first one, which is assumed to be the label.
    # y_train will contain the first column which is the label
    X_train = reference.iloc [:, 1:]
    y_train = reference.iloc [:, 0]
    
    # Feature scaling using MinMaxScaler to scale the features to a range of [0, 1].
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    
    # Define SVM parameter
    param_grid = {
    'C': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
    }
    
    # Define F1 as scoring metrics
    f1_scorer = make_scorer(f1_score)
    
    # Initialize Hyperparameter Tuning
    grid_search = GridSearchCV(model, param_grid, scoring=f1_scorer, cv=20, verbose = 1, n_jobs = -1)
    grid_search.fit (X_train, y_train)
    
    # Find out best parameter
    final_param = grid_search.best_params_
    print (f"Final parameter: {final_param}")
    
    # Find out best model
    final_model = grid_search.best_estimator_
    
    # Save the best model 
    joblib.dump (final_model, 'SVM_model.pkl')

if __name__ == "__main__":
    print("Starting the training script...")
    main()




