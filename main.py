# %% 
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import median_filter
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from scipy import stats

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
    """Process all image data for a single organoid with image alignment."""
    for condition_dir in [f.path for f in os.scandir(parent_dir) if f.is_dir()]:
        condition_name = os.path.basename(condition_dir)
        print(f"Processing condition: {condition_name}")

        try:
            power = extract_power(condition_name)
        except ValueError as e:
            print(f"Skipping condition {condition_name}: {e}")
            continue

        nadh_file = next((f for f in os.listdir(condition_dir) if f.endswith('745nm.tif')), None)
        fad_file = next((f for f in os.listdir(condition_dir) if f.endswith('860nm.tif')), None)

        if nadh_file and fad_file:
            nadh_stack = tifffile.imread(os.path.join(condition_dir, nadh_file))
            fad_stack = tifffile.imread(os.path.join(condition_dir, fad_file))

            # Compute max intensity projections
            nadh_max = np.max(nadh_stack, axis=0)
            fad_max = np.max(fad_stack, axis=0)

            # Align images based on max intensity projections
            shift, _, _ = phase_cross_correlation(nadh_max, fad_max, upsample_factor=100)
            aligned_fad_stack = shift(fad_stack, shift=(0, shift[0], shift[1]), mode='constant', cval=0)

            # Process aligned images
            processed_nadh = process_image(nadh_stack, power, bg_coords)
            processed_fad = process_image(aligned_fad_stack, power, bg_coords)

            # Save processed images
            output_dir = os.path.join('processed', os.path.basename(parent_dir), condition_name)
            os.makedirs(output_dir, exist_ok=True)
            tifffile.imwrite(os.path.join(output_dir, f'{condition_name}_745nm.tif'), processed_nadh)
            tifffile.imwrite(os.path.join(output_dir, f'{condition_name}_860nm.tif'), processed_fad)

        else:
            print(f"Skipping condition {condition_name}: Missing NADH or FAD image")
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

# %% Redox Ratio Calculations - Functions
def calculate_redox_ratio(parent_dir, roi_coords=(0, 0), roi_size=(50, 50)):
    """_summary_

    Args:
        parent_dir (_type_): folder with nadh and fad tiffs only. should be 'processed/Organoid 1/[xxx]mw_[yyy]nm.tif'
        roi_coords (tuple, optional): y_start and x_start. Defaults to (0, 0).
        roi_size (tuple, optional): y_size and x_size. Defaults to (50, 50).
        
        if roi_size == (0, 0), ROI will be full image 
    """
    # Loop through files in parent_dir
    for file in os.listdir(parent_dir):
        if file.endswith('745nm.tif'):
            nadh_image_stack = tifffile.imread(os.path.join(parent_dir, file))
        elif file.endswith('860nm.tif'):
            fad_image_stack = tifffile.imread(os.path.join(parent_dir, file))

    if 'nadh_image_stack' not in locals() or 'fad_image_stack' not in locals():
        raise ValueError("Could not find both NADH and FAD images in the directory.")
    
    if roi_size == (0, 0):
        roi_size = nadh_image_stack.shape[1:]  # Set to image dimensions (y, x)
        roi_coords = (0, 0)  # Start from the top-left corner when using full image
    
    rois = [(slice(roi_coords[0], roi_coords[0]+roi_size[0]), slice(roi_coords[1], roi_coords[1]+roi_size[1]))]
    
    nadh_max = np.max(nadh_image_stack, axis=0) 
    fad_max = np.max(fad_image_stack, axis=0)
    # Draw ROI and determine redox ratio
    # Create visualization of ROIs on first image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.imshow(nadh_max, cmap='gray')
    ax1.set_title('NADH')
    ax1.axis('image')

    ax2.imshow(fad_max, cmap='gray')
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

    file_name = parent_dir.split('/')
    plt.tight_layout()
    os.makedirs('processed/max_intensities', exist_ok=True)
    plt.savefig(f'processed/max_intensities/{file_name[1]}_with_ROIs.png')
    plt.show()

    RR_arr = []
    
    for j in range(len(nadh_image_stack)):
        
        nadh_roi = nadh_image_stack[j][roi_coords[0]:roi_coords[0]+roi_size[0], roi_coords[1]:roi_coords[1]+roi_size[1]]    
        fad_roi = fad_image_stack[j][roi_coords[0]:roi_coords[0]+roi_size[0], roi_coords[1]:roi_coords[1]+roi_size[1]]
        
        ratio = np.divide(fad_roi.astype(float), (nadh_roi.astype(float) +fad_roi.astype(float)), out=np.zeros_like(nadh_roi, dtype=float), where=fad_roi+nadh_roi!=0)
        
        
        RR_arr.append(np.mean(ratio)) # taking the mean here could result in data loss contributing to less power for the t test.
        
    print(f'{parent_dir[:-1]} - Avg. Redox Ratio: {round(np.mean(RR_arr), 2)}')
    # RR_arr returned contains the redox ratio at each correspending depth; array of numpy float values
    return RR_arr

o1_arr = calculate_redox_ratio('processed/Organoid1/', roi_coords = (180, 280))
o2_arr = calculate_redox_ratio('processed/Organoid2/', roi_coords = (290, 190))
DMSO_arr = calculate_redox_ratio('processed/Organoid_DMSO_treated/', roi_coords = (200, 200))
DOX_arr = calculate_redox_ratio('processed/Organoid_DOX_treated/', roi_coords = (280, 240))

# DMSO_arr_test = calculate_redox_ratio('processed/Organoid_DMSO_treated/', roi_coords = (0, 0), roi_size=(1, 1))
# DOX_arr_test = calculate_redox_ratio('processed/Organoid_DOX_treated/', roi_coords = (0, 0), roi_size=(1, 1))


# %% T Testing
# Analysis and plotting
## 2. compare control and DOX treated organoids
### Can do 2 sample t-test to compare between the 2 groups
print(f'org1 vs org2: {stats.ttest_ind(o1_arr, o2_arr)}')
print(f'DMSO vs DOX: {stats.ttest_ind(DMSO_arr, DOX_arr)}')

# %%
## 3. Qualitative visualization
def create_redox_ratio_colormap(parent_dir, roi_coords=(100, 100)):
    try:
        # Read NADH and FAD image stacks
        nadh_file = next(f for f in os.listdir(parent_dir) if f.endswith('745nm.tif'))
        fad_file = next(f for f in os.listdir(parent_dir) if f.endswith('860nm.tif'))
        
        nadh_stack = tifffile.imread(os.path.join(parent_dir, nadh_file))
        fad_stack = tifffile.imread(os.path.join(parent_dir, fad_file))

        # Max z-projection
        nadh_max = np.max(nadh_stack, axis=0)
        fad_max = np.max(fad_stack, axis=0)

        # Calculate redox ratio
        redox_ratio = np.divide(fad_max.astype(float), (nadh_max.astype(float) + fad_max.astype(float)),
                                out=np.zeros_like(nadh_max, dtype=float), where=fad_max+nadh_max!=0)

        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # Plot NADH max projection
        ax1.imshow(nadh_max, cmap='gray')
        ax1.set_title('NADH Max Projection')
        ax1.axis('off')

        # Plot FAD max projection
        ax2.imshow(fad_max, cmap='gray')
        ax2.set_title('FAD Max Projection')
        ax2.axis('off')

        # Plot redox ratio colormap
        im = ax3.imshow(redox_ratio, cmap='coolwarm', vmin=0, vmax=1)
        ax3.set_title('Redox Ratio Colormap')
        ax3.axis('off')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label('Redox Ratio')
        
        file_name = parent_dir[:-1].split('/')
        plt.suptitle(f'Analysis - {os.path.basename(parent_dir)}')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f'./{file_name[0]}/Redox_Ratios/{file_name[1]}_redox_ratio_colormap.png')
        plt.show()

        return redox_ratio
    except FileNotFoundError:
        print(f"Error: Required files not found in directory {parent_dir}")
        return None

# Generate colormaps for each condition
org1_redox = create_redox_ratio_colormap('processed/Organoid1/')
org2_redox = create_redox_ratio_colormap('processed/Organoid2/')
dmso_redox = create_redox_ratio_colormap('processed/Organoid_DMSO_treated/')
dox_redox = create_redox_ratio_colormap('processed/Organoid_DOX_treated/')

# %% TODO

# Image Processing
## 1. custom background coordinates for each organoid for background subtraction - top left corner 