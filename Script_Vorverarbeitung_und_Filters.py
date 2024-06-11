#!/usr/bin/env python
# coding: utf-8

# # Filters and Vorverarbeitungen

# ## Install all the packages

# In[ ]:


# Install all the packages
get_ipython().system('pip install opencv-python-headless pillow matplotlib')
get_ipython().system('pip install scikit-image')
import scipy
from scipy.integrate import quad
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ## Load the Image and save them in array

# In[ ]:


##TEST, code for read only on image
# # ***Define the image path***
# image_path = 'bild.jpg'
# # ***Check if the image can be read***
# if not os.path.exists(image_path):
#     raise FileNotFoundError(f"The image {image_path} does not exist")
# # ***Load the image in grayscale mode***
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

##Read and save multiple images in an array
import cv2
import numpy as np
import os

#Define a 
def load_images_from_folder(folder_path):
    """
    Load all images from the specified folder and store them in a numpy array.

    Parameters:
    folder_path (str): Path to the folder containing images.

    Returns:
    numpy.ndarray: Array containing all the images.
    """
    images = [] # Create an empty list to store the images
    for filename in os.listdir(folder_path):
        # Construct the full path to the image file
        img_path = os.path.join(folder_path, filename)
        # Read the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Check if the image was successfully loaded
        if img is not None:
            images.append(img)
        else:
            print(f"Error loading image {img_path}")
    
    # Convert the list of images to a numpy array
    images_array = np.array(images)
    return images_array

# ##Use Example
# # Use for example
# folder_path = 'folder_path' #Enter your folder of images here!
# images_array = load_images_from_folder(folder_path)

# # Show the numer of read images
# print(f"Loaded {len(images_array)} images.")


# ## Unsharp Masking Filter, Laplace Edge Enhancement Filter and Sobel Edge Enhancement

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import data
import scipy

def unsharp_mask(image, ksize=9, sigma=10.0, alpha=1.5, beta=-0.5):
    """
    Apply unsharp masking to enhance the edges of the image.

    Parameters:
    image (numpy.ndarray): The input grayscale image.
    ksize (int): Size of the Gaussian kernel. Default is 9.
                 # Range: odd numbers greater than 0, e.g., 3, 5, 7, 9, 11
    sigma (float): Standard deviation for Gaussian kernel. Default is 10.0.
                   # Range: positive floats, e.g., 1.0, 2.5, 10.0, 20.0
    alpha (float): Weight of the original image. Default is 1.5.
                   # Range: positive floats, e.g., 1.0, 1.5, 2.0
    beta (float): Weight of the Gaussian blurred image. Default is -0.5.
                  # Range: negative floats, e.g., -0.5, -1.0

    Returns:
    edge_enhanced (numpy.ndarray): The edge-enhanced image.
    """
    gaussian = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    edge_enhanced = cv2.addWeighted(image, alpha, gaussian, beta, 0)
    return edge_enhanced

def laplacian_edge_enhancement(image, ddepth=cv2.CV_64F, ksize=3):
    """
    Apply Laplacian edge enhancement filter to the image.

    Parameters:
    image (numpy.ndarray): The input grayscale image.
    ddepth (int): Desired depth of the destination image. Default is cv2.CV_64F.
                  # Range: cv2.CV_8U, cv2.CV_16U, cv2.CV_64F, etc.
    ksize (int): Aperture size used to compute the second-derivative filters. Default is 3.
                 # Range: 1, 3, 5, etc. (odd numbers)

    Returns:
    edge_enhanced (numpy.ndarray): The edge-enhanced image.
    """
    laplacian = cv2.Laplacian(image, ddepth, ksize=ksize)
    edge_enhanced = cv2.convertScaleAbs(laplacian)
    return edge_enhanced

def sobel_edge_enhancement(image, ddepth=cv2.CV_64F, ksize=3, alpha=0.5, beta=0.5):
    """
    Apply Sobel edge enhancement filter to the image.

    Parameters:
    image (numpy.ndarray): The input grayscale image.
    ddepth (int): Desired depth of the destination image. Default is cv2.CV_64F.
                  # Range: cv2.CV_8U, cv2.CV_16U, cv2.CV_64F, etc.
    ksize (int): Size of the extended Sobel kernel. Default is 3.
                 # Range: 1, 3, 5, etc. (odd numbers)
    alpha (float): Weight of the gradient in x direction. Default is 0.5.
                   # Range: floats between 0 and 1, e.g., 0.3, 0.5, 0.7
    beta (float): Weight of the gradient in y direction. Default is 0.5.
                  # Range: floats between 0 and 1, e.g., 0.3, 0.5, 0.7

    Returns:
    edge_enhanced (numpy.ndarray): The edge-enhanced image.
    """
    grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=ksize)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edge_enhanced = cv2.addWeighted(abs_grad_x, alpha, abs_grad_y, beta, 0)
    return edge_enhanced


# # Usage Example
# # Load images from a folder
# folder_path = '/work/cropped leg images/' #Enter your folder path here!
# images_array = load_images_from_folder(folder_path)

# # Define the folder to save processed images
# output_folder = '/work/Test1/' #Enter the output folder path, if it doesn't exist, it will create one.
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Process each image in the array
# for i, img in enumerate(images_array):
#     unsharp_img = unsharp_mask(img)
#     laplacian_img = laplacian_edge_enhancement(img)
#     sobel_img = sobel_edge_enhancement(img)
    
#     # Save the processed images to the output folder
#     cv2.imwrite(os.path.join(output_folder, f'unsharp_img_{i}.png'), unsharp_img)
#     cv2.imwrite(os.path.join(output_folder, f'laplacian_img_{i}.png'), laplacian_img)
#     cv2.imwrite(os.path.join(output_folder, f'sobel_img_{i}.png'), sobel_img)

# #Show the processed images
#     # # Optionally, display the processed images
#     # plt.imshow(unsharp_img, cmap='gray')
#     # plt.show()


# ## Implementation of the Filters and show the Comparision of 3 Different Filters

# In[ ]:


# # Display the original and enhanced images
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title('Original Image')
# axes[0].axis('off')

# axes[1].imshow(sobel_enhanced, cmap='gray')
# axes[1].set_title('Sobel Edge Enhanced')
# axes[1].axis('off')

# axes[2].imshow(laplacian_enhanced, cmap='gray')
# axes[2].set_title('Laplacian Edge Enhanced')
# axes[2].axis('off')

# axes[3].imshow(unsharp_enhanced, cmap='gray')
# axes[3].set_title('Unsharp Mask Enhanced')
# axes[3].axis('off')

# plt.tight_layout()
# plt.show()


# ## Median Filter

# In[ ]:


import cv2
from scipy.ndimage import median_filter
import numpy as np
import os

# Load images from a folder function
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            print(f"Error loading image {img_path}")
    images_array = np.array(images)
    return images_array

# Median Filter Function
def apply_median_filter(image, filter_size=3):
    """
    Apply a median filter to the input image.

    Parameters:
    image (numpy.ndarray): Input image.
    filter_size (int): Size of the median filter. Default is 3.

    Returns:
    numpy.ndarray: Median filtered image.
    """
    median_filtered = median_filter(image, size=filter_size)
    return median_filtered

# Usage Example
# Load images from a folder
folder_path = '/work/cropped leg images/' # Enter your folder path here!
images_array = load_images_from_folder(folder_path)

# Define the folder to save processed images
output_folder = '/work/Medianfilter/' # Enter the output folder path, if it doesn't exist, it will create one.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the array
for i, img in enumerate(images_array):
    median_img = apply_median_filter(img)  # Use the correct function name
    # Save the processed images to the output folder
    cv2.imwrite(os.path.join(output_folder, f'median_img_{i}.png'), median_img)

# For visualization (commented out as requested)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.title("Median Filtered Image")
# plt.imshow(filtered_image, cmap='gray')
# plt.axis('off')
# plt.show()


# ## Laplace Filter

# In[ ]:


import cv2
import numpy as np

def apply_laplacian_filter(image):
    """
    Apply the Laplacian filter to the input image.

    Parameters:
    image (numpy.ndarray): Input grayscale image.
    ksize (int): Aperture size used to compute the second-derivative filters. Default is 5.
                 # Range: Odd numbers greater than 1, e.g., 3, 5, 7, 9.
    scale (float): Optional scale factor for the computed Laplacian values. Default is 0.2.
                   # Range: Positive floats, e.g., 0.1, 0.2, 1.0, 2.0.
    delta (float): Optional delta value added to the results before storing them in the output. Default is 0.
                   # Range: Floats, e.g., -10, 0, 10.

    Returns:
    numpy.ndarray: Laplacian filtered image.
    """
    # Apply the Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=5, scale=0.2, delta=0)

    # Convert the result back to uint8
    laplacian = cv2.convertScaleAbs(laplacian)

    return laplacian

## Example usage
## Load the image (assuming grayscale)
#image_path = 'bild.jpg'
#image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply the Laplacian filter
#laplacian_image = apply_laplacian_filter(image)

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title("Orginal Ultraschallbild")
# plt.axis('off')
# plt.subplot(1,2,2)
# plt.imshow(laplacian, cmap='gray')
# plt.title('Laplace Graustufenbild')
# plt.axis('off')

# plt.show()


# 

# ## Sharpen the Image using Sharpen Kernel

# In[ ]:


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Load images from a folder function
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            print(f"Error loading image {img_path}")
    images_array = np.array(images)
    return images_array

# Sharpen Image Function
def sharpen_image(image, kernel=None):
    """
    Apply a sharpening filter to the input image.

    Parameters:
    image (numpy.ndarray): Input image.
    kernel (numpy.ndarray): Sharpening kernel. Default is a 3x3 kernel.
                            # Range: Any custom kernel, e.g., np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]).

    Returns:
    numpy.ndarray: Sharpened image.
    """
    # Create the default sharpening kernel if none is provided
    if kernel is None:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Sharpen the image
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Usage Example
# Load images from a folder
folder_path = '/work/cropped leg images/'  # Enter your folder path here!
images_array = load_images_from_folder(folder_path)

# Define the folder to save processed images
output_folder = '/work/SharpenedImages/'  # Enter the output folder path, if it doesn't exist, it will create one.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the array
for i, img in enumerate(images_array):
    sharpened_img = sharpen_image(img)  # Use the correct function name
    # Save the processed images to the output folder
    cv2.imwrite(os.path.join(output_folder, f'sharpened_img_{i}.png'), sharpened_img)

# For visualization (commented out as requested)
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(images_array[0], cmap='gray')
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.title("Sharpened Image")
# plt.imshow(sharpened_img, cmap='gray')
# plt.axis('off')
# plt.show()



# ## Wiener Filter

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
import cv2
import os

# Load images from a folder function
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            print(f"Error loading image {img_path}")
    images_array = np.array(images)
    return images_array

# Wiener Filter Function
def apply_wiener_filter(image, mysize=(5, 5), noise_est=0.1):
    """
    Apply the Wiener filter to the input image.

    Parameters:
    image (numpy.ndarray): Input grayscale image.
    mysize (tuple): Size of the Wiener filter window. Default is (5, 5).
                    # Range: Tuple of two integers, e.g., (3, 3), (5, 5), (7, 7).
    noise_est (float): Estimate of the noise power. Default is 0.1.
                       # Range: Positive float, e.g., 0.01, 0.1, 1.0.

    Returns:
    numpy.ndarray: Wiener filtered image.
    """
    # Apply Wiener filter
    wiener_filtered = wiener(image, mysize=mysize, noise=noise_est)

    # Clip the filtered result to [0, 255] and convert to uint8 type
    wiener_filtered_uint8 = np.clip(wiener_filtered, 0, 255).astype(np.uint8)

    return wiener_filtered_uint8

# Usage Example
# Load images from a folder
folder_path = '/work/cropped leg images/'  # Enter your folder path here!
images_array = load_images_from_folder(folder_path)

# Define the folder to save processed images
output_folder = '/work/Wienerfilter/'  # Enter the output folder path, if it doesn't exist, it will create one.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the array
for i, img in enumerate(images_array):
    wiener_img = apply_wiener_filter(img)  # Use the correct function name
    # Save the processed images to the output folder
    cv2.imwrite(os.path.join(output_folder, f'wiener_img_{i}.png'), wiener_img)

# For visualization (commented out as requested)
# plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(images_array[0], cmap='gray')
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.title("Wiener Filtered Image")
# plt.imshow(wiener_img, cmap='gray')
# plt.axis('off')
# plt.show()


# In[ ]:


import numpy as np
from scipy.signal import wiener
import cv2

def apply_wiener_filter(image_path, mysize, noise_est):
    """
    Load an image, apply a Wiener filter, and return the filtered image.

    Parameters:
    image_path (str): Path to the input image.
    mysize (tuple): Size of the Wiener filter window.
    noise_est (float): Estimate of the noise power.

    Returns:
    numpy.ndarray: Wiener filtered image.
    """
    # Load the image in grayscale
    image = cv2.imread('bild.jpg', cv2.IMREAD_GRAYSCALE)

# Ensure the image is read correctly
if image is None:
    print("Error: Could not load image")
else:
    # Apply the Wiener filter with specified parameters
    mysize = (5, 5)
    noise_est = 0.1

    # Apply Wiener filter
    wiener_filtered = wiener(image, mysize=mysize, noise=noise_est)

    # Clip the filtered result to [0, 255] and convert to uint8 type
    wiener_filtered_uint8 = np.clip(wiener_filtered, 0, 255).astype(np.uint8)

# Example usage
image_path = '/work/2023-09-08 16.47.54.jpg'
wiener_filtered_image = apply_wiener_filter(image_path)


# ##  Adjusting Contrast and Brightness

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('/work/2023-09-08 16.47.54.jpg')

# Adjust contrast and brightness
alpha = 1.5 # Contrast factor
beta = 10   # Brightness offset

adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Display the original and adjusted images
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Contrast and Brightness Adjusted")
plt.imshow(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))

plt.show()


# ## Gamma Correlation

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
#image = cv2.imread('/work/Ultraschall_Armen/geschneidete Bilder/auto_cropped_2024-05-14 17.13.58.jpg')
image = cv2.imread('/work/2023-09-08 16.47.54.jpg')
# Gamma Correction function
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Apply Gamma Correction
gamma = 1.0 # Gamma value
gamma_corrected_image = adjust_gamma(image, gamma=gamma)

# Display the original and gamma corrected images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Gamma Corrected")
plt.imshow(cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2RGB))

plt.show()


# ## High and Low Frequency Filter

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the image in grayscale mode
image_path = '/work/2023-09-08 16.47.54.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not load image")
else:
    # Get the image size
    rows, cols = image.shape

    # Fourier Transformation of the Image
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a low pass filter (remove high frequencies)
    crow, ccol = rows // 2 , cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 30  # Radius
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1

    # Apply the Low pass filter
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Create a high pass filter (remove low frequencies)
    mask_high = np.ones((rows, cols, 2), np.uint8)
    mask_high[mask_area] = 0

    # Apply the high pass filter
    fshift_high = dft_shift * mask_high
    f_ishift_high = np.fft.ifftshift(fshift_high)
    img_back_high = cv2.idft(f_ishift_high)
    img_back_high = cv2.magnitude(img_back_high[:, :, 0], img_back_high[:, :, 1])

    # Show the Consequence
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_back, cmap='gray')
    plt.title('Low Pass Filtered Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_back_high, cmap='gray')
    plt.title('High Pass Filtered Image')
    plt.axis('off')

    plt.show()


# ## Gauß Filter

# In[ ]:


image = cv2.imread('Bild.jpg', cv2.IMREAD_GRAYSCALE) #Füge für Bild.jpg den gewünschten Pfad aus der Drive aus
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
gaussian_blur_uint8 = cv2.convertScaleAbs(gaussian_blur)
plt.figure(figsize=(12, 6))
plt.imshow(gaussian_blur_uint8, cmap='gray')
plt.title('Gauß-gefiltertes Bild vor CLAHE-Anwendung')
plt.axis('off')
plt.show()


# 

# 

# ## CLAHE

# In[ ]:


image = cv2.imread('Bild.jpg', cv2.IMREAD_GRAYSCALE)#Füge für Bild.jpg den gewünschten Pfad aus der Drive aus
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 2)
plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE Graustufenbild')
plt.axis('off')

plt.show()


# 

# # Gauß Filter & CLAHE

# 

# In[ ]:


image = cv2.imread('Bild.jpg', cv2.IMREAD_GRAYSCALE)#Füge für Bild.jpg den gewünschten Pfad aus der Drive aus
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)
gaussian_blur = cv2.GaussianBlur(clahe_image, (5, 5), 0)
gaussian_blur_uint8 = cv2.convertScaleAbs(gaussian_blur)

plt.figure(figsize=(12, 6))
# Gauß-gefiltertes Bild anzeigen
plt.subplot(1, 2, 1)
plt.imshow(gaussian_blur_uint8, cmap='gray')
plt.title('Gauß-gefiltertes Bild')
plt.axis('off')

# Bild nach CLAHE-Anwendung anzeigen
plt.subplot(1, 2, 2)
plt.imshow(clahe_image, cmap='gray')
plt.title('Bild nach CLAHE-Anwendung')
plt.axis('off')

plt.show()

