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

def process_image(image_stack, power, bg_coords=(0, 0)):
    """Process a single image stack."""
    # Flatten the image stack into average values in a 2D array
    y, x = image_stack.shape[1:]  # Get y and x dimensions
    averaged_image = np.mean(image_stack, axis=0)

    # Perform background subtraction
    bg_value = calculate_background(averaged_image, roi_start=bg_coords)
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


def process_organoid_data(parent_dir, bg_coords=(0, 0)):
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
                image_stacks.append(process_image(image_stack, power, bg_coords))
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
    process_organoid_data(organoid_dir, bg_coords=(0,50)) # Example coordinates


# Perform redox ratio calculations
def calculate_redox_ratio(dir_745, dir_860, output_dir):
    """Calculates the redox ratio between two image stacks."""

    try:
        stack_745 = tifffile.imread(dir_745)
        stack_860 = tifffile.imread(dir_860)

        # Check if dimensions match
        if stack_745.shape != stack_860.shape:
            raise ValueError("Image stack dimensions do not match for ratio calculation.")

        # Calculate redox ratio (745nm / 860nm)
        redox_ratio = np.divide(stack_745.astype(np.float32), stack_860.astype(np.float32), out=np.zeros_like(stack_745, dtype=np.float32), where=stack_860!=0)

        # Save the redox ratio image
        output_path = os.path.join(output_dir, f"{os.path.basename(dir_745).replace('_745nm', '_ratio')}")
        tifffile.imwrite(output_path, redox_ratio)
        print(f"Saved redox ratio to: {output_path}")

    except FileNotFoundError:
        print(f"Could not find matching 745nm and 860nm files for ratio calculation.")
    except ValueError as e:
        print(f"Error during ratio calculation: {e}")



def process_redox_ratios(processed_dir):
    """Process redox ratios for all organoids and conditions."""
    os.makedirs("redox_ratios", exist_ok=True)
    for filename in os.listdir(processed_dir):
        if filename.endswith(".tif"):
            match_745 = re.search(r"_(\d+)mw_745nm\.tif", filename)
            if match_745:
                power_745 = match_745.group(1)
                base_filename = filename.replace(f"_{power_745}mw_745nm.tif", "")
                file_860 = f"{base_filename}_{power_745}mw_860nm.tif" # Assumes same power for both wavelengths

                file_path_745 = os.path.join(processed_dir, filename)
                file_path_860 = os.path.join(processed_dir, file_860)

                calculate_redox_ratio(file_path_745, file_path_860, "redox_ratios")


# Analysis and plotting
## 1. averaging across depth

## 2. compare control and DOX treated organoids

## 3. qualitative visualization
process_redox_ratios('processed')

# Analysis and plotting
## 1. averaging across depth

## 2. compare control and DOX treated organoids

## 3. qualitative visualization
