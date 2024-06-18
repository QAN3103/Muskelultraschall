#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Install all the packages
get_ipython().system('apt-get update')
get_ipython().system('apt-get install -y libgl1-mesa-glx')
#!pip install opencv-python-headless pillow matplotlib
get_ipython().system('pip install opencv-contrib-python')
get_ipython().system('pip install scikit-image')
get_ipython().system('pip install glrlm')
import scipy
from scipy.integrate import quad
#Import all the packages
import os
import shutil
import pandas as pd
import cv2
import numpy as np
import scipy
from skimage.feature import graycomatrix, graycoprops
import fnmatch
import matplotlib.pyplot as plt
from skimage import data
from glrlm import GLRLM


# In[ ]:


import pandas as pd
import cv2
import os
import numpy as np
import scipy
from skimage.feature import graycomatrix, graycoprops
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
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

def calculate_hist(hist):
    """
    Calculate statistical properties of a histogram.

    Parameters:
    hist (numpy.ndarray): The histogram of an image.

    Returns:
    mean_hist (float): The mean of the histogram.
    median_hist (float): The median of the histogram.
    std_hist (float): The standard deviation of the histogram.
    skewness_hist (float): The skewness of the histogram.
    kurtosis_hist (float): The kurtosis of the histogram.
    """
    mean_hist = float(np.mean(hist))
    median_hist = float(np.median(hist))
    std_hist = float(np.std(hist))
    skewness_hist = float(scipy.stats.skew(hist, axis=0, bias=True)[0])
    kurtosis_hist = float(scipy.stats.kurtosis(hist, axis=0, fisher=True, bias=True)[0])
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
    # Define paths
    csv_path = '/work/Train/train.csv'
    root_folder = '/work/Train/'  # Root directory containing the images
    output_csv = '/work/Train/Features.csv'

    # Read image IDs from CSV
    image_ids, original_df = read_image_ids(csv_path)

    # Process images and save results
    process_images(image_ids, root_folder, output_csv, original_df)

if __name__ == "__main__":
    main()

