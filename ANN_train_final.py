"""
Muscle Atrophy Detection with Artificial Neural Network

This script is intended to develop and train an ANN model using Keras for the purpose of muscle atrophy detection. The process consists of the following steps:
1. Importing the train feature table.
2. Separating feature matrix, defined as X_train, and label vector, defined as y_train.
3. Feature scaling using MinMaxScaler between (0,1). 
4. Initiating ANN Model consisting of one hidden layer with ReLu as activation function. 
5. Defining callbacks for model training.
6. Performing Hyperparameter-Tuning using GridSearchCV and printing out the best parameters
7. Train the model using the best Hyperparameters and Cross Validation 
8. Saving the best model

Authors:
- Quỳnh Anh Nguyễn
- Heyi Wang
- Lea Grün
- Dilan Mohammadi

Functions:
- create_model(): Constructs the ANN model with specified architecture.
- main(): Initiate the training process.

Requirements:
- Python 3.x
- Libraries: TensorFlow, Keras, Scikit-learn, Pandas, NumPy, SciKeras

"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l1_l2

def create_model(input_shape):
    """
    Creates and compiles a Sequential neural network model with the specified learning rate.
    
    Parameters:
        input_shape (int): The shape of the input data.
    
    Returns:
        model (tf.keras.Model): A compiled Keras model ready for training.
    """
    
    # Initialize the Sequential model
    model = Sequential([
        # Input layer: specifies the shape of the input data
        Input(shape=(input_shape,)),
        
        # First hidden layer: Dense layer with 16 units, ReLU activation, and L1-L2 regularization
        Dense(16, activation='relu', kernel_regularizer=l1_l2(0.01)),
        
        # Batch Normalization layer: normalizes the inputs to the next layer
        BatchNormalization(),
        
        # Output layer: Dense layer with 1 unit and sigmoid activation for binary classification
        Dense(1, activation='sigmoid')
    ])
    
    # Define the Adam optimizer with the specified learning rate
    adam = Adam(learning_rate= 0.001)
    
    # Compile the model with binary cross-entropy loss and accuracy metric
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    """
    Main function to execute the training process.
    """
    # Import training feature table. 
    # Note: The file path must be updated to the appropriate location before use.
    reference = pd.read_csv(r'C:\Users\Quynh Anh\Muskelultraschall\features_train.csv')

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

    # Convert the scaled feature dataframe and label series to numpy arrays for model training.
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Define Callbacks

    # ModelCheckpoint: Save only the best model based on 'val_loss'
    # - 'filepath': Path to save the model file. Must be changed before use
    model_checkpoint = ModelCheckpoint(
        filepath=r'C:\Users\Quynh Anh\Muskelultraschall\ANN_model.keras', 
        monitor='val_loss', 
        save_best_only=True, 
        mode='min', 
        verbose=1)

    # ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving, in this case 'val_loss'
    # If 'val_loss' does not improve after 10 epochs, learning rate will be reduced by factor 0.1 but will not be reduced to lower than 0.0000001
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.1, 
        patience=10, 
        min_lr=0.0000001, 
        verbose=1, 
        mode='min')

    # EarlyStopping: Stop training when a monitored metric has stopped improving, in this case 'val_loss'
    # If 'val_loss' does not improve after 20 epochs, training will be stopped
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        verbose=1, 
        restore_best_weights=True)

    # List of callbacks to be passed to the model during training.
    callbacks = [model_checkpoint, reduce_lr, early_stopping]
    
    #  KerasClassifier-Wrapper
    model = KerasClassifier(build_fn=create_model (X_train.shape[1]), verbose=1)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    # Define the parameter grid for Hyperparameter-Tuning using GridSearchCV
    param_grid = {
    'classifier__batch_size': [8, 16, 32, 64, 128],
    'classifier__epochs': [25, 50, 75, 100]
}
    
    # Define the GridSearchCV
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1', error_score='raise')
    
    # Run GridSearchCV to find the best hyperparameters
    grid_search.fit(X_train, y_train, classifier__callbacks=callbacks)
    
    # Output the best parameters found by GridSearchCV
    final_param = grid_search.best_params_
    print(f"Final parameters: {final_param}")
    
    # Initialize KFold. In this case, a 10-fold Cross Validation was used
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X_train):
        # Split the data
        X1, X2 = X_train[train_index], X_train[test_index]
        y1, y2 = y_train[train_index], y_train[test_index]

        # Create a new model instance
        model = create_model(X_train.shape[1])

        # Train the model
        history = model.fit(X1, y1, epochs= final_param['classifier__epochs'], batch_size=final_param['classifier__batch_size'], callbacks=[early_stopping, model_checkpoint, reduce_lr], verbose=1, validation_data=(X2, y2))
        
if __name__ == "__main__":
    print("Starting the training script...")
    main()






