import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
# import matplotlib.animation as animation
# from PIL import Image

# 1. Read the TIFF stack and show information
image_stack = imread("media/Organoid 1/353mw_745nm/0um_353mw_745.TIF")

# Get the dimensions of the stack
z, y, x = image_stack.shape
print(f"Image stack dimensions: {z} x {y} x {x}")

## Flatten the image stack into average values in a 2D array
newArray = np.zeros((x, y))
for i in range(y):
    for j in range(x):
        newArray[i][j] = np.mean(image_stack[:, i, j])
    
plt.imshow(newArray, cmap='gray')
plt.axis('off')  # Remove axes for cleaner display
plt.show()

# 2. background subtraction

# 3. alignment adjestment between 2 channels

# 4. noise reduction with median filter

# 5. intensity normalization


# Perform redox ratio calculations


# Analysis and plotting
## 1. averaging across depth

## 2. compare control and DOX treated organoids

## 3. qualitative visualization
