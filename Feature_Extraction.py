#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import data
import scipy


# In[9]:


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


# In[10]:


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


# In[8]:


def calculate_glcm (image, distance, angle, level):
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
    """
    glcm = graycomatrix(image, distance, angle, level, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    return contrast, dissimilarity, homogeneity, energy, correlation


# In[11]:


#kantendetektion


# In[ ]:




