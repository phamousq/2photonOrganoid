import matplotlib.pyplot as plt
import numpy as np
import tifffile
import os
from pathlib import Path
# import matplotlib.animation as animation
# from PIL import Image


# 1. get average image
def process_image(file_path):
    image_stack = tifffile.imread(file_path)
    
    # Get the dimensions of the stack
    z, y, x = image_stack.shape
    
    # Flatten the image stack into average values in a 2D array
    newArray = np.zeros((x, y))
    for i in range(y):
        for j in range(x):
            newArray[i][j] = np.mean(image_stack[:, i, j])
    
    # # Display average image
    # plt.imshow(newArray, cmap='gray')
    # plt.axis('off')  # Remove axes for cleaner display
    # plt.show()
    
    ## Data Stats
    # print(f"Image stack dimensions: {z} x {y} x {x}")
    # print(f'max value: {np.amax(newArray)}')
    normalizeto16bit = ((newArray - np.amin(newArray)) / (np.amax(newArray) - np.amin(newArray)) * 65535).astype(np.uint16)
    return normalizeto16bit

# Create processed directory if it doesn't exist
os.makedirs('processed', exist_ok=True)

stackOrganoid = []
pathRoot = 'media'

for parent in [f.path for f in os.scandir(pathRoot) if f.is_dir()]:
    organoid_name = os.path.basename(parent)  # Get organoid folder name
    print(f"Processing organoid: {organoid_name}")
    
    for child in [f.path for f in os.scandir(parent) if f.is_dir()]:
        condition_name = os.path.basename(child)  # Get condition folder name
        print(f"Processing condition: {condition_name}")
        
        # Process all TIF files in this condition
        for filename in os.listdir(child):
            if filename.endswith(('.tif', '.TIF')):
                file_path = os.path.join(child, filename)
                processed_image = process_image(file_path)
                stackOrganoid.append(processed_image)
        
        if stackOrganoid:  # Only save if we have processed images
            # Create output filename based on organoid and condition
            output_filename = f"{organoid_name}_{condition_name}.tif"
            output_path = os.path.join('processed', output_filename)
            
            # Convert list to numpy array and save
            stack_array = np.array(stackOrganoid)
            print(f"Saving processed stack to: {output_path}")
            print(f"Stack shape: {stack_array.shape}")
            tifffile.imwrite(output_path, stack_array)
        
        # Clear the stack for next condition
        stackOrganoid = []


# 2. background subtraction
## need to account for when subtraction will be negative

# 3. alignment adjestment between 2 channels

# 4. noise reduction with median filter

# 5. intensity normalization


# Perform redox ratio calculations


# Analysis and plotting
## 1. averaging across depth

## 2. compare control and DOX treated organoids

## 3. qualitative visualization
