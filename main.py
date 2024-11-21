import matplotlib.pyplot as plt
import numpy as np
import tifffile
import os
from pathlib import Path
from scipy.ndimage import median_filter

# 1. get average image
def calculate_background(image, roi_start=(0, 0), roi_size=(50, 50)):
    """Calculate background value from a region of interest (ROI)"""
    roi_end = (roi_start[0] + roi_size[0], roi_start[1] + roi_size[1])
    roi = image[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1]]
    return np.mean(roi)

def process_image(file_path):
    image_stack = tifffile.imread(file_path)
    
    # Get the dimensions of the stack
    z, y, x = image_stack.shape
    
    # Flatten the image stack into average values in a 2D array
    newArray = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            newArray[i][j] = np.mean(image_stack[:, i, j])
    
    # Perform background subtraction
    bg_value = calculate_background(newArray)
    # Custom ROI:
    # bg_value = calculate_background(newArray, roi_start=(100, 100), roi_size=(100, 100))
    newArray = np.clip(newArray - bg_value, 0, None)  # Subtract background, clip negative values to 0
    
    # Apply median filter for noise reduction
    # Use a 3x3 kernel for moderate noise reduction while preserving edges
    filtered_array = median_filter(newArray, size=3)
    
    ## Data Stats
    # print(f"Image stack dimensions: {z} x {y} x {x}")
    # print(f'max value: {np.amax(filtered_array)}')
    normalizeto16bit = ((filtered_array - np.amin(filtered_array)) / (np.amax(filtered_array) - np.amin(filtered_array)) * 65535).astype(np.uint16)
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
        
        # Get all TIF files and sort them numerically
        tif_files = [f for f in os.listdir(child) if f.endswith(('.tif', '.TIF'))]
        # Sort files based on the numeric value in the filename
        sorted_files = sorted(tif_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
        
        # Process files in sorted order
        for filename in sorted_files:
            # print(filename)
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

# 3. alignment adjestment between 2 channels

# 4. noise reduction with median filter

# 5. intensity normalization


# Perform redox ratio calculations


# Analysis and plotting
## 1. averaging across depth

## 2. compare control and DOX treated organoids

## 3. qualitative visualization
