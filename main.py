# %% 
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import median_filter

# %% 
POWER_TRANSMISSION = 0.20  # 20% power transmission at sample
def extract_power(folder_name):
    """Extract laser power from folder name."""
    power_match = re.search(r'(\d+)mw', folder_name)
    if power_match:
        power_mw = float(power_match.group(1))
        extracted_power = round(power_mw * POWER_TRANSMISSION, 1)
        print(f"Extracted power: {extracted_power} mw")
        return extracted_power
    else:
        raise ValueError(f"Could not extract power from folder name: {folder_name}")

def calculate_background(image, roi_start=(0, 0), roi_size=(50, 50)):
    """
    Calculate background value from a region of interest (ROI). 
    this is a 50x50 pixel square starting roi_start.
    """ 
    roi_end = (roi_start[0] + roi_size[0], roi_start[1] + roi_size[1])
    roi = image[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1]]
    return np.mean(roi)

# %%
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

# a = process_image('media/Organoid 1_353mw_745nm.tif', 353)
# plt.imshow(a, cmap='gray')
# plt.show()

# %%
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
            os.makedirs(f'processed/{os.path.basename(parent_dir)}', exist_ok=True)
            output_filename = f"{os.path.basename(parent_dir)}/{condition_name}.tif"
            output_path = os.path.join('processed', output_filename)
            
            try:
                tifffile.imwrite(output_path, np.array(image_stacks))
                print(f"Saved processed stack to {output_path}")
            except Exception as e:
                print(f"Error saving processed stack to {output_path}: {e}")


# %% Create processed images for each organoid
# Create processed directory if it doesn't exist
os.makedirs('processed', exist_ok=True)
pathRoot = 'media'

for organoid_dir in [f.path for f in os.scandir(pathRoot) if f.is_dir()]:
    organoid_name = os.path.basename(organoid_dir)
    print(f"Processing organoid: {organoid_name}")
    process_organoid_data(organoid_dir, bg_coords=(0,50)) # Example coordinates

# %% Redox Ratio Calculations
def calculate_redox_ratio(parent_dir, roi_coords=[100, 100]):
    """_summary_

    Args:
        parent_dir (_type_): folder with nadh and fad tiffs only. should be 'processed/Organoid 1/[xxx]mw_[yyy]nm.tif'
        roi_coords (tuple, optional): y_start and x_start. Defaults to (100, 100).
    """
    # Loop through files in parent_dir
    for file in os.listdir(parent_dir):
        if file.endswith('745nm.tif'):
            nadh_image = tifffile.imread(os.path.join(parent_dir, file))[0]
        elif file.endswith('860nm.tif'):
            fad_image = tifffile.imread(os.path.join(parent_dir, file))[0]

    if 'nadh_image' not in locals() or 'fad_image' not in locals():
        raise ValueError("Could not find both NADH and FAD images in the directory.")

    rois = [(slice(roi_coords[0], roi_coords[0]+50), slice(roi_coords[1], roi_coords[1]+50))]
    # Draw ROI and determine redox ratio
    # Create visualization of ROIs on first image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.imshow(nadh_image, cmap='gray')
    ax1.set_title('NADH')
    ax1.axis('image')

    ax2.imshow(fad_image, cmap='gray')
    ax2.set_title('FAD')
    ax2.axis('image')

    # Highlight ROIs
    colors = ['red']
    for (roi, color) in zip(rois, colors):
        y_slice, x_slice = roi
        rect = plt.Rectangle((x_slice.start, y_slice.start), 
                            x_slice.stop - x_slice.start, 
                            y_slice.stop - y_slice.start,
                            fill=False, color=color, linewidth=2)
        ax1.add_patch(rect)
        # Add text label near the ROI
        ax1.text(x_slice.start, y_slice.start-5, 'ROI', color=color, 
                fontsize=10, fontweight='bold')

        rect2 = plt.Rectangle((x_slice.start, y_slice.start), 
                            x_slice.stop - x_slice.start, 
                            y_slice.stop - y_slice.start,
                            fill=False, color=color, linewidth=2)
        ax2.add_patch(rect2)
        # Add text label near the ROI
        ax2.text(x_slice.start, y_slice.start-5, 'ROI', color=color, 
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

    nadh_roi = nadh_image[roi_y:roi_y+50, roi_x:roi_x+50]
    fad_roi = fad_image[roi_y:roi_y+50, roi_x:roi_x+50]

    ratio = np.divide(fad_roi.astype(float), (nadh_roi.astype(float) +fad_roi.astype(float)), out=np.zeros_like(nadh_roi, dtype=float), where=fad_roi+nadh_roi!=0)
    print(f'{parent_dir[:-1]} Redox Ratio: {round(np.mean(ratio), 2)}')

calculate_redox_ratio('processed/Organoid1/', roi_coords = [80, 280])
calculate_redox_ratio('processed/Organoid2/', roi_coords = [200, 400])
calculate_redox_ratio('processed/Organoid_DMSO_treated/', roi_coords = [200, 200])
calculate_redox_ratio('processed/Organoid_DOX_treated/', roi_coords = [120, 320])


# %% TODO

# Image Processing
# 1. custom background coordinates for each organoid 
# 2. alignment adjustments between channels


# Analysis and plotting
## Redox Ratio
## averaging across depth

## 2. compare control and DOX treated organoids
## 3. qualitative visualization
# %%
