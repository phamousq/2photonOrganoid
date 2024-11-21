import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import median_filter

# Constants
POWER_TRANSMISSION = 0.20  # 20% power transmission at sample

def extract_power(folder_name):
    """Extract laser power from folder name."""
    power_match = re.search(r'(\d+)mw', folder_name)
    if power_match:
        power_mw = float(power_match.group(1))
        return power_mw * POWER_TRANSMISSION
    else:
        raise ValueError(f"Could not extract power from folder name: {folder_name}")

def calculate_background(image, roi_start=(0, 0), roi_size=(50, 50)):
    """Calculate background value from a region of interest (ROI)."""
    roi_end = (roi_start[0] + roi_size[0], roi_start[1] + roi_size[1])
    roi = image[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1]]
    return np.mean(roi)

def process_image(image_stack, power):
    """Process a single image stack."""
    # Flatten the image stack into average values in a 2D array
    y, x = image_stack.shape[1:]  # Get y and x dimensions
    averaged_image = np.mean(image_stack, axis=0)

    # Perform background subtraction
    bg_value = calculate_background(averaged_image)
    background_subtracted = np.clip(averaged_image - bg_value, 0, None)

    # Apply median filter for noise reduction
    filtered_image = median_filter(background_subtracted, size=3)

    # Normalize by power squared
    power_normalized = filtered_image / (power ** 2)

    # Normalize to 16-bit
    normalized_16bit = ((power_normalized - np.amin(power_normalized)) /
                       (np.amax(power_normalized) - np.amin(power_normalized)) * 65535).astype(np.uint16)
    
    # plt.imshow(normalized_16bit, cmap='gray')
    # plt.show()
    
    return normalized_16bit


def process_organoid_data(parent_dir):
    """Process all image data for a single organoid."""

    for condition_dir in [f.path for f in os.scandir(parent_dir) if f.is_dir()]:
        condition_name = os.path.basename(condition_dir)
        print(f"Processing condition: {condition_name}")

        try:
            power = extract_power(condition_name)
        except ValueError as e:
            print(f"Skipping condition {condition_name}: {e}")
            continue

        image_stacks = []
        tif_files = sorted([f for f in os.listdir(condition_dir) if f.endswith(('.tif', '.TIF'))],
                           key=lambda x: int(''.join(filter(str.isdigit, x))))

        for filename in tif_files:
            file_path = os.path.join(condition_dir, filename)
            try:
                image_stack = tifffile.imread(file_path)
                image_stacks.append(process_image(image_stack, power))
            except Exception as e:  # Catch any errors during file reading/processing
                print(f"Error processing file {filename}: {e}")
                continue  # Skip to the next file

        if image_stacks:
            output_filename = f"{os.path.basename(parent_dir)}_{condition_name}.tif"
            output_path = os.path.join('processed', output_filename)
            
            try:
                tifffile.imwrite(output_path, np.array(image_stacks))
                print(f"Saved processed stack to {output_path}")
            except Exception as e:
                print(f"Error saving processed stack to {output_path}: {e}")

# Create processed directory if it doesn't exist
os.makedirs('processed', exist_ok=True)

pathRoot = 'media'

for organoid_dir in [f.path for f in os.scandir(pathRoot) if f.is_dir()]:
    organoid_name = os.path.basename(organoid_dir)
    print(f"Processing organoid: {organoid_name}")
    process_organoid_data(organoid_dir)


# TODO
# Perform redox ratio calculations


# Analysis and plotting
## 1. averaging across depth

## 2. compare control and DOX treated organoids

## 3. qualitative visualization

# TODO 
# Perform redox ratio calculations


# Analysis and plotting
## 1. averaging across depth

## 2. compare control and DOX treated organoids

## 3. qualitative visualization
