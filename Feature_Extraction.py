#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import data
import scipy


# In[ ]:


def create_hist (image):
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


# In[ ]:


def calculate_hist (hist):
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
    mean_hist = np.mean (hist)
    median_hist = np.median (hist)
    std_hist = np.std (hist)
    skewness_hist = scipy.stats.skew(hist, axis=0, bias=True)
    kurtosis_hist = scipy.stats.kurtosis(hist, axis=0, fisher=True, bias=True)
    return mean_hist, median_hist, std_hist, skewness_hist, kurtosis_hist


# In[ ]:


import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread

def calculate_glcm(image, distance, angle, level):
    #Example: glcm = graycomatrix(patient_1, distances=[100], angles=[0], levels=256, symmetric=True, normed=True)
    """
    Calculate the Gray Level Co-occurrence Matrix (GLCM) and its properties.

    Parameters:
    image (numpy.ndarray): The input grayscale image.
    distance (list of int): The list of pixel pair distance offsets.
    angle (list of float): The list of pixel pair angles in radians.
    level (int): The number of gray levels in the image.

    Returns:
    contrast (numpy.ndarray): The contrast of the GLCM.
    dissimilarity (numpy.ndarray): The dissimilarity of the GLCM.
    homogeneity (numpy.ndarray): The homogeneity of the GLCM.
    energy (numpy.ndarray): The energy of the GLCM.
    correlation (numpy.ndarray): The correlation of the GLCM.
    entropy (numpy.ndarray): The entropy of the GLCM.
    """
    # Compute GLCM
    glcm = graycomatrix(image, distances=distance, angles=angle, levels=level, symmetric=True, normed=True)
    
    # Compute GLCM properties
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)), axis=(0, 1))
    
    return contrast, dissimilarity, homogeneity, energy, correlation, entropy

# # Usage Example
# #First read the images

# # Ensure the image is in 8-bit integer format
# image = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image

# # Define distances and angles
# distances = [1]
# angles = [0]

# # Calculate GLCM properties
# contrast, dissimilarity, homogeneity, energy, correlation, entropy = calculate_glcm(image, distances, angles, level=256)

# # Display the results
# angles_degrees = [0]
# for i, angle in enumerate(angles_degrees):
#     print(f"Angle {angle} degrees:")
#     print(f"    Contrast: {contrast[0, i]}")
#     print(f"    Dissimilarity: {dissimilarity[0, i]}")
#     print(f"    Homogeneity: {homogeneity[0, i]}")
#     print(f"    Energy: {energy[0, i]}")
#     print(f"    Correlation: {correlation[0, i]}")
#     print(f"    Entropy: {entropy[i]}")

