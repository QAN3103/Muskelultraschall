"""
Muscle Atrophy Detection with Logistic Regression

This script is intended to develop and train a Logistic Regression (LoRe) model for the purpose of muscle atrophy detection. The process includes the following steps:
1. Importing the train feature table.
2. Separating feature matrix, defined as X_train, and label vector, defined as y_train.
3. Feature scaling using MinMaxScaler between (0,1). 
4. Performing GridSearch to find the best parameter combination of C - Solver - Penalty. With each parameter combination, a model is trained using cross validation and evaluated based on Accuracy Score. 
5. Saving the best model and printing out the best parameter combination.
6. Performing Recursive Feature Ablation with Cross-Validation (RFECV) to find the optimal number of features. The model is evaluated using the F1-score for each feature subset.
7. Printing the optimal number of features and the selected features.
8. Saving a plot of the number of features vs. cross-validation F1 score.

Authors:
- Quỳnh Anh Nguyễn
- Heyi Wang
- Lea Grün
- Dilan Mohammadi

Functions:
- main(): Initiate the training process.

Requirements:
- Python 3.x
- Libraries: Scikit-learn, Pandas, Joblib, Matplotlib

"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, f1_score
import matplotlib.pyplot as plt
import joblib
from sklearn import linear_model
import pandas as pd

def main():
    """
    Main function to execute the training process.
    """
    # Import the training feature table. 
    # Note: The file path must be updated to the appropriate location before use.
    reference = pd.read_csv('C:/Users/Quynh Anh/Muskelultraschall/features_train.csv')

    # Remove the 'Image_ID' column as it is not needed for training.
    reference.drop('Image_ID', axis='columns', inplace=True)

    # Create training data by separating features and labels.
    # X_train will contain all columns except the first one, which is assumed to be the label.
    # y_train will contain the first column which is the label.
    X_train = reference.iloc[:, 1:]
    y_train = reference.iloc[:, 0]

    # Feature scaling using MinMaxScaler to scale the features to a range of [0, 1].
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    # Create a logistic regression model with specified parameters.
    # The model will use L2 regularization, and run for a maximum of 200 iterations. Both classes have the same weight
    model = linear_model.LogisticRegression(penalty='l2', class_weight='balanced', max_iter=200)     

    # Define the cross-validation strategy
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],       # Regularization strength
    'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],  # Optimization algorithms
    'penalty': [None, 'l1', 'l2']                # Regularization type
    }

    # Initialize GridSearchCV with the logistic regression model, parameter grid, and cross-validation strategy
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kf,
        scoring='accuracy',      # Use accuracy to evaluate model performance
        verbose=1                # Set verbosity to see detailed progress
    )

    # Run GridSearchCV to find the best hyperparameters
    grid_search.fit(X_train, y_train)

    # Output the best parameters found by GridSearchCV
    final_param = grid_search.best_params_
    print(f"Final parameters: {final_param}")

    # Save the best model to a file using joblib
    final_model = grid_search.best_estimator_
    joblib.dump(final_model, 'Lo_Re_model.pkl')
    
     # Initialize the best model with the best hyperparameters found
    best_model = linear_model.LogisticRegression(penalty=final_param['penalty'], class_weight= 'balanced', max_iter = 200, C = final_param['C'], solver = final_param['solver'])
    
    # Define the scorer for RFECV using F1 score
    scorer = make_scorer(f1_score)
    
    # Perform RFECV to select the optimal number of features
    rfecv = RFECV(estimator=best_model, step=1, cv=KFold(5), scoring=scorer)

    # Fit RFECV
    rfecv.fit(X_train, y_train)
    
    # Plot number of features vs. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross-validation F1 score")
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'], marker='o', color = 'lightgreen')
    plt.grid(True)

    # Get the optimal number of features
    optimal_num_features = rfecv.n_features_
    selected_features = X_train.columns[rfecv.support_]

    print(f"Optimal number of features: {optimal_num_features}")
    print(f"Selected features: {selected_features}")
    plt.savefig ('Feature_Importance_LogReg.png')

if __name__ == "__main__":
    print("Starting the training script...")
    main()



