#!/usr/bin/env python
# coding: utf-8

# # Install DL Track US

# In[ ]:


# use pip to install DL Track US
#pip install DL-Track-US


# ## Install all the Packages

# In[ ]:


# Install all the packages
get_ipython().system('pip install opencv-python-headless pillow matplotlib')
import scipy
from scipy.integrate import quad
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ## Install Scikit-Image

# In[ ]:


#install scikit-image
# Update pip
get_ipython().system('pip install -U pip')
# Install scikit-image
get_ipython().system('pip install scikit-image')
#To sum up. For skimage version 0.19 and above, this is the right import and version:
#from skimage.feature import graycomatrix, graycoprops
import skimage
print(skimage.__file__)
import os
skimage_path = skimage.__file__.rsplit('/', 1)[0]
feature_path = os.path.join(skimage_path, 'feature')
print(os.listdir(feature_path))



# 

# ## Implementation of Features

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread

# Load the grayscale image
image = imread('/work/cropped images with individual size/auto_cropped_2024-05-21 16.14.41.jpg')

# Ensure the image is in 8-bit integer format
image = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image

# Compute the GLCM
distances = [1]  # Distance for GLCM
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Directions for GLCM
glcm = graycomatrix(image, distances, angles, levels=256, symmetric=True, normed=True)

# Compute Haralick features
contrast = graycoprops(glcm, 'contrast')
correlation = graycoprops(glcm, 'correlation')
energy = graycoprops(glcm, 'energy')
homogeneity = graycoprops(glcm, 'homogeneity')
entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))  # Custom entropy calculation

# Display the results
print(f"Contrast: {contrast}")
print(f"Correlation: {correlation}")
print(f"Energy: {energy}")
print(f"Homogeneity: {homogeneity}")
print(f"Entropy: {entropy}")

# Visualize the image and its GLCM
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(glcm[:, :, 0, 0], cmap='gray')  # Show GLCM for one direction
plt.title('GLCM (0 degrees)')
plt.axis('off')

plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread

# Load the grayscale image
image = imread('/work/cropped images with individual size/auto_cropped_2024-05-21 16.14.41.jpg')

# Ensure the image is in 8-bit integer format
image = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image

# Define distances and angles
distances = [1]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Compute GLCM
glcm = graycomatrix(image, distances, angles, levels=256, symmetric=True, normed=True)

# Compute Haralick features
contrast = graycoprops(glcm, 'contrast')[0]
correlation = graycoprops(glcm, 'correlation')[0]
energy = graycoprops(glcm, 'energy')[0]
homogeneity = graycoprops(glcm, 'homogeneity')[0]
entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))

# Display the results
angles_degrees = [0, 45, 90, 135]
for i, angle in enumerate(angles_degrees):
    print(f"Angle {angle} degrees:")
    print(f"    Contrast: {contrast[i]}")
    print(f"    Correlation: {correlation[i]}")
    print(f"    Energy: {energy[i]}")
    print(f"    Homogeneity: {homogeneity[i]}")

print(f"Entropy: {entropy}")

# Visualize the image and its GLCM
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(glcm[:, :, 0, 0], cmap='gray')
plt.title('GLCM (0 degrees)')
plt.axis('off')

plt.show()


# In[ ]:


# Plotting the features
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(angles_degrees, contrast, marker='o')
plt.title('Contrast vs Angles')
plt.xlabel('Angle (degrees)')
plt.ylabel('Contrast')

plt.subplot(2, 2, 2)
plt.plot(angles_degrees, correlation, marker='o')
plt.title('Correlation vs Angles')
plt.xlabel('Angle (degrees)')
plt.ylabel('Correlation')

plt.subplot(2, 2, 3)
plt.plot(angles_degrees, energy, marker='o')
plt.title('Energy vs Angles')
plt.xlabel('Angle (degrees)')
plt.ylabel('Energy')

plt.subplot(2, 2, 4)
plt.plot(angles_degrees, homogeneity, marker='o')
plt.title('Homogeneity vs Angles')
plt.xlabel('Angle (degrees)')
plt.ylabel('Homogeneity')

plt.tight_layout()
plt.show()


# 

# In[ ]:


#Try Entropy
import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
image = imread('/work/cropped images with individual size/auto_cropped_2024-05-21 16.14.41.jpg')

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4), sharex=True, sharey=True)

img0 = ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title("Image")
ax0.axis("off")
fig.colorbar(img0, ax=ax0)

img1 = ax1.imshow(entropy(image, disk(5)), cmap='gray')
ax1.set_title("Entropy")
ax1.axis("off")
fig.colorbar(img1, ax=ax1)

fig.tight_layout()

plt.show()


# In[ ]:


import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread

def calculate_glcm(image, distance, angle, level):
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

