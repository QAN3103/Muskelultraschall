"""
Script to Calculate Features

This script demonstrates the Feature Engineering proces. The process includes the following steps:

1. Reading image IDs from a reference CSV file containing all Image IDs and labels.
2. Splitting the data set into Training Set and Test Set.
3. Finding the corresponding bone and muscle image segments in the designated root directory.
4. Creating histograms for the images and calculating statistical features from the histograms.
6. Computing texture features using Gray Level Co-occurrence Matrix (GLCM) and Gray Level Run Length Matrix (GLRLM).
7. Saving the calculated features into new CSV files for both the Training Set and the Test Set.

Author: 
- Quỳnh Anh Nguyễn
- Heyi Wang
- Dilan Mohammadi
- Lea Grün

Functions:
- read_image_ids(csv_path): Reads image IDs from a CSV file.
- create_hist(image): Creates a histogram for a grayscale image.
- calculate_hist(hist): Calculates statistical properties of a histogram.
- calculate_glcm(image, distances, angles, levels): Calculates the GLCM and its properties.
- calculate_glrlm(image): Calculates GLRLM features from input image.
- find_image(image_name, root_folder, image_type, suffix): Finds and loads a grayscale image from a root folder with a given suffix.
- process_images(image_ids, root_folder, output_csv, original_df): Processes images based on the image IDs, calculates features, and saves results 
  to a new CSV.
- main(): Orchestrates the entire workflow of the script, including defining file paths, reading image IDs from a CSV file, processing images to 
extract features, and saving these features to an output CSV file.

Requirements: 
- Python 3.x
- Libraries: OpenCV (cv2), Pandas, NumPy, scikit-image, glrlm, Scikit-learn
"""

import os
import pandas as pd
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from glrlm import GLRLM


def read_image_ids(csv_path):
    """
    Read image IDs from a CSV file.
    """
    df = pd.read_csv(csv_path)
    return df['Image_ID'].tolist(), df

def create_hist(image):
    """
    Create a histogram for a grayscale image.

    Parameters:
    image (numpy.ndarray): The input grayscale image.

    Returns:
    hist (numpy.ndarray): The histogram of the image.
    """
    # Calculate the histogram for the grayscale image
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

def calculate_hist(hist):
    """
    Calculate statistical features of a histogram.

    Parameters:
    hist (numpy.ndarray): The histogram of an image.

    Returns:
    mean_hist (float): The mean of the histogram.
    median_hist (float): The median of the histogram.
    std_hist (float): The standard deviation of the histogram.
    skewness_hist (float): The skewness of the histogram.
    kurtosis_hist (float): The kurtosis of the histogram.
    """
    # Normalize the histogram
    hist_normalized = hist.ravel() / hist.sum()

    # Calculate the mean 
    mean_hist = np.sum(hist_normalized * np.arange(256))

    # Calculate the standard deviation 
    std_hist = np.sqrt(np.sum(hist_normalized * (np.arange(256) - mean_hist)**2))

    # Calculate the skewness 
    skewness_hist = np.sum(hist_normalized * ((np.arange(256) - mean_hist) ** 3)) / (std_hist ** 3)

    # Calculate the kurtosis 
    kurtosis_hist = np.sum(hist_normalized * ((np.arange(256) - mean_hist) ** 4)) / (std_hist ** 4)

    # Calculate the cumulative histogram to find the median
    cumulative_hist = np.cumsum(hist_normalized)

    # Find the median pixel value
    median_hist = np.searchsorted(cumulative_hist, 0.5)
    return mean_hist, median_hist, std_hist, skewness_hist, kurtosis_hist

def calculate_glcm(image, distances, angles, levels):
    """
    Calculate the Gray Level Co-occurrence Matrix (GLCM) and its properties.

    Parameters:
    image (numpy.ndarray): The input grayscale image.
    distance (list of int): The list of pixel pair distance offsets.
    angle (list of float): The list of pixel pair angles in radians.
    level (int): The number of gray levels in the image.

    Returns:
    contrast (float): The contrast of the GLCM.
    dissimilarity (float): The dissimilarity of the GLCM.
    homogeneity (float): The homogeneity of the GLCM.
    energy (float): The energy of the GLCM.
    correlation (float): The correlation of the GLCM.
    entropy (float): The entropy of the GLCM.
    """
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    dissimilarity = float(graycoprops(glcm, 'dissimilarity')[0, 0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    energy = float(graycoprops(glcm, 'energy')[0, 0])
    correlation = float(graycoprops(glcm, 'correlation')[0, 0])
    entropy = float(-np.sum(glcm * np.log2(glcm + (glcm == 0))))
    
    return contrast, dissimilarity, homogeneity, energy, correlation, entropy

def calculate_glrlm(image):
    """
    Calculate Gray Level Run Length Matrix (GLRLM) features from an input image.
    
    This function reads an input image in grayscale, computes the GLRLM features,
    and returns several features derived from the GLRLM.

    Parameters:
    -----------
    image : str
        Path to the input image file.

    Returns:
    --------
    tuple
        A tuple containing the following GLRLM features:
        - SRE (Short Run Emphasis)
        - LRE (Long Run Emphasis)
        - GLU (Gray Level Uniformity)
        - RLU (Run Length Uniformity)
        - RPC (Run Percentage)

    Example:
    --------
    >>> sre, lre, glu, rlu, rpc = calculate_glrlm('path/to/image.png')
    >>> print(f"SRE: {sre}, LRE: {lre}, GLU: {glu}, RLU: {rlu}, RPC: {rpc}")

    """
    app = GLRLM()
    features = app.get_features(image, 8)
    return float(features.SRE), float(features.LRE), float(features.GLU), float(features.RLU), float(features.RPC)

def find_image(image_name, root_folder, image_type, suffix):
    """
    Find and load a grayscale image from a root folder with a given suffix.

    Parameters:
    image_name (str): The base name of the image file (e.g., 'TU1R_1').
    root_folder (str): The root directory to search for the image file.
    image_type (str): The file extension of the image (e.g., '.png' or '.jpg').
    suffix (str): The suffix to add to the image name (e.g., '_bone' or '_muscle').

    Returns:
    image (numpy.ndarray): The loaded grayscale image. Returns None if the image could not be found or loaded.
    """
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file == image_name + suffix + image_type:
                absolute_path = os.path.join(root, file)
                image = cv2.imread(absolute_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Error: Could not load image from {absolute_path}. Please check the file path and file format.")
                return image
    print(f"Error: File not found for {image_name + suffix + image_type} in {root_folder}")
    return None

def process_images(image_ids, root_folder, output_csv, original_df):
    """
    Process images based on the image IDs, calculate features, and save results to a new CSV.
    """
    bone_results = []
    muscle_results = []

    for image_id in image_ids:
        # Find and load bone image
        bone_image = find_image(image_id, root_folder, '.jpg', '_bone')
        if bone_image is not None:
            hist = create_hist(bone_image)
            hist_features = calculate_hist(hist)
            glcm_features = calculate_glcm(bone_image, distances=[1], angles=[0], levels=256)
            glrlm_features = calculate_glrlm(bone_image)
            result = [image_id] + list(hist_features) + list(glcm_features) + list(glrlm_features)
            bone_results.append(result)

        # Find and load muscle image
        muscle_image = find_image(image_id, root_folder, '.jpg', '_muscle')
        if muscle_image is not None:
            hist = create_hist(muscle_image)
            hist_features = calculate_hist(hist)
            glcm_features = calculate_glcm(muscle_image, distances=[1], angles=[0], levels=256)
            glrlm_features = calculate_glrlm(muscle_image)
            result = [image_id] + list(hist_features) + list(glcm_features) + list(glrlm_features)
            muscle_results.append(result)

    # Define columns with appropriate prefixes for bone and muscle features
    bone_columns = ["Image_ID", "bone_Mean_Hist", "bone_Median_Hist", "bone_Std_Hist", "bone_Skewness_Hist", "bone_Kurtosis_Hist", 
                    "bone_Contrast", "bone_Dissimilarity", "bone_Homogeneity", "bone_Energy", "bone_Correlation", "bone_Entropy",
                    "bone_SRE", "bone_LRE", "bone_GLU", "bone_RLU", "bone_RPC"]
    muscle_columns = ["Image_ID", "muscle_Mean_Hist", "muscle_Median_Hist", "muscle_Std_Hist", "muscle_Skewness_Hist", "muscle_Kurtosis_Hist", 
                      "muscle_Contrast", "muscle_Dissimilarity", "muscle_Homogeneity", "muscle_Energy", "muscle_Correlation", "muscle_Entropy",
                      "muscle_SRE", "muscle_LRE", "muscle_GLU", "muscle_RLU", "muscle_RPC"]

    # Create DataFrames for bone and muscle features
    bone_results_df = pd.DataFrame(bone_results, columns=bone_columns)
    muscle_results_df = pd.DataFrame(muscle_results, columns=muscle_columns)

    # Merge the original dataframe with the new results on 'Image_ID'
    final_df = pd.merge(original_df, bone_results_df, on="Image_ID", how="left")
    final_df = pd.merge(final_df, muscle_results_df, on="Image_ID", how="left")
    
    final_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

def main():
    """
    Main function to execute the preprocessing, model construction, and training process.
    """
    # Read the REFERENCE.CSV file. The directory must be changed before use
    reference = pd.read_csv('C:/Users/Quynh Anh/Muskelultraschall/REFERENCE.csv')
    
    # Extract the 'Image_ID' and 'Krank/Gesund' columns
    X = reference ['Image_ID']
    y = reference ['Krank/Gesund']
    
    # Split the data into training and test sets with stratification, ratio Training Set:Test Set of 80:20
    x_train, x_test, y_train, y_test = train_test_split (X, y, test_size=0.2, stratify=y)
    
    # Combine the training features and labels into a single DataFrame
    train_set = pd.concat ((pd.DataFrame(x_train), pd.DataFrame (y_train)), axis = 1)
    
    # Combine the test features and labels into a single DataFrame
    test_set = pd.concat ((pd.DataFrame(x_test), pd.DataFrame (y_test)), axis = 1)
    
    # Save the training and test sets to separate CSV files. The directory to save the created file must be changed before use
    train_set.to_csv ('C:/Users/Quynh Anh/Muskelultraschall/train.csv', index = False)
    test_set.to_csv ('C:/Users/Quynh Anh/Muskelultraschall/test.csv', index = False)
    
    # Define paths for the CSV file, root folder containing images, and output CSV file.
    csv_path_train = 'train.csv' # file path of train csv table
    csv_path_test = 'test.csv' # file path of test csv table
    
    root_folder = r'C:\Users\Quynh Anh\Muskelultraschall'  # Root directory containing the images
    
    output_csv_train = 'Features_train.csv' # define the output of the calculated features table of training set.
    output_csv_test = 'Features_test.csv' # define the output of the calculated features table of test set.

    # Read image IDs from CSV
    image_ids_train, original_df_train = read_image_ids(csv_path_train)
    image_ids_test, original_df_test = read_image_ids(csv_path_test)

    # Process images and save results
    process_images(image_ids_train, root_folder, output_csv_train, original_df_train)
    process_images(image_ids_test, root_folder, output_csv_test, original_df_test)

#call main in Jupyternotebook
if __name__ == "__main__":
    print("Starting the script...")
    main()





