#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import cv2
import os
# Read the REFERENCE.CSV file. Absolute Path must be changed before use
reference = pd.read_csv('C:/Users/Quynh Anh/Muskelultraschall/data/REFERENCE.csv')


# In[9]:


image_id = reference ["Image_ID"]


# In[10]:


def get_bone_image (image_name, path, image_type):
    """
    Load a grayscale bone image from the specified path.

    Parameters:
    image_name (str): The base name of the image file (e.g., 'TU2N_1').
    path (str): The directory path where the image file is located (e.g., 'C:/Users/Quynh Anh/Muskelultraschall/data_example/').
    image_type (str): The file extension of the image (e.g., '.png' or '.jpg').

    Returns:
    image (numpy.ndarray): The loaded grayscale image. Returns None if the image could not be loaded.
    """ 
    # Construct the absolute path to the image file
    absolute_path = os.path.join(path, image_name + '_bone' + image_type)
    
    # Verify that the file exists
    if not os.path.exists(absolute_path):
        print(f"Error: File not found at {absolute_path}")
        return None

    # Load the image in grayscale mode
    image = cv2.imread(absolute_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image from {absolute_path}. Please check the file path and file format.")
    
    return image
    


# In[11]:


#patient_1 = get_bone_image (image_id[0], 'C:/Users/Quynh Anh/Muskelultraschall/data_example/', '.png')


# In[12]:


def get_muscle_image (image_name, path, image_type):
    """
    Load a grayscale muscle image from the specified path.

    Parameters:
    image_name (str): The base name of the image file (e.g., 'TU2N_1').
    path (str): The directory path where the image file is located (e.g., 'C:/Users/Quynh Anh/Muskelultraschall/data_example/').
    image_type (str): The file extension of the image (e.g., '.png' or '.jpg').

    Returns:
    image (numpy.ndarray): The loaded grayscale image. Returns None if the image could not be loaded.
    """
    # Construct the absolute path to the image file
    absolute_path = os.path.join(path, image_name + '_muscle' + image_type)
    
    # Verify that the file exists
    if not os.path.exists(absolute_path):
        print(f"Error: File not found at {absolute_path}")
        return None

    # Load the image in grayscale mode
    image = cv2.imread(absolute_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image from {absolute_path}. Please check the file path and file format.")
    
    return image


# In[13]:


jupyter nbconvert Load_from_reference.ipynb --to python


# In[14]:


ipython nbconvert Load_from_reference.ipynb --to script


# In[ ]:




