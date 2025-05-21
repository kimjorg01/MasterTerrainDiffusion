import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_psd_2d(image_path):
    """
    Loads a grayscale image and calculates its 2D Power Spectral Density.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: The 2D PSD (magnitude squared of centered FFT),
                       or None if the image cannot be processed.
    """
    try:
        img = Image.open(image_path).convert('L') 
        img_array = np.array(img, dtype=np.float32) 

        
        fft_result = np.fft.fft2(img_array)

        
        fft_shifted = np.fft.fftshift(fft_result)

        
        psd_2d = np.abs(fft_shifted)**2

        return psd_2d

    except FileNotFoundError:
        logging.warning(f"File not found: {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

def save_psd_visualization(psd_array, output_path, title="Average 2D PSD"):
    """Saves a visualization of the 2D PSD using a log scale."""
    if psd_array is None:
        logging.warning(f"Cannot save visualization for None PSD array: {output_path}")
        return
    try:
        plt.figure(figsize=(6, 6))
        
        
        min_val = np.min(psd_array[psd_array > 0]) if np.any(psd_array > 0) else 1e-5
        plt.imshow(psd_array, cmap='magma', norm=LogNorm(vmin=min_val, vmax=psd_array.max()))
        plt.colorbar(label='Log Power')
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(output_path, bbox_inches='tight')
        plt.close() 
        logging.info(f"Saved PSD visualization to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save PSD visualization {output_path}: {e}")


def main(args):
    image_folder_path = args.image_dir

    if not os.path.isdir(image_folder_path):
        logging.error(f"Provided image directory not found or is not a directory: {image_folder_path}")
        print(f"Error: Image directory '{image_folder_path}' not found.")
        return

    
    folder_label = os.path.basename(os.path.normpath(image_folder_path))

    
    if args.output_dir_psd:
        output_psd_dir = args.output_dir_psd
    else:
        
        output_psd_dir = os.path.join(image_folder_path, "psd_analysis")

    os.makedirs(output_psd_dir, exist_ok=True)
    logging.info(f"PSD results for '{folder_label}' will be saved in: {output_psd_dir}")

    image_files = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
    
    logging.info(f"Scanning for images in {image_folder_path}...")
    for f_name in os.listdir(image_folder_path):
        if f_name.lower().endswith(supported_extensions):
            image_files.append(os.path.join(image_folder_path, f_name))

    if not image_files:
        logging.warning(f"No image files found in {image_folder_path}")
        print(f"No images found in '{image_folder_path}'. Ensure images have extensions like {supported_extensions}.")
        return

    logging.info(f"Found {len(image_files)} images to process in {image_folder_path}.")

    
    
    
    psd_accumulator = {"sum": None, "count": 0, "expected_shape": None}

    logging.info(f"Calculating 2D PSD for images in '{folder_label}'...")
    for image_path in tqdm(image_files, desc=f"Processing Images in '{folder_label}' for PSD"):
        psd_2d = calculate_psd_2d(image_path)

        if psd_2d is not None:
            if psd_accumulator["sum"] is None:
                
                psd_accumulator["sum"] = np.zeros_like(psd_2d, dtype=np.float64)
                psd_accumulator["expected_shape"] = psd_2d.shape
            
            if psd_2d.shape == psd_accumulator["expected_shape"]:
                psd_accumulator["sum"] += psd_2d
                psd_accumulator["count"] += 1
            else:
                logging.warning(f"Shape mismatch for {os.path.basename(image_path)} ({psd_2d.shape}) vs expected ({psd_accumulator['expected_shape']}). Skipping this image for averaging.")

    
    logging.info(f"Calculating and saving average PSD for '{folder_label}'...")
    print(f"\n--- Average PSD Calculation Summary for: {folder_label} ---")

    count = psd_accumulator["count"]
    psd_sum = psd_accumulator["sum"]
    avg_psd = None

    if count > 0 and psd_sum is not None:
        avg_psd = psd_sum / count
        print(f"  Calculated average PSD based on {count} images (Expected shape: {psd_accumulator['expected_shape']}).")

        
        npy_filename = f"{folder_label}_average_psd.npy"
        npy_filepath = os.path.join(output_psd_dir, npy_filename)
        try:
            np.save(npy_filepath, avg_psd)
            print(f"  Saved average PSD data to: {npy_filepath}")
            logging.info(f"Saved average PSD .npy for {folder_label} to {npy_filepath}")
        except Exception as e:
            logging.error(f"Failed to save average PSD .npy for {folder_label}: {e}")
            print(f"  Error saving average PSD data for {folder_label}.")

        
        if args.save_visualizations:
            vis_filename = f"{folder_label}_average_psd.png"
            vis_filepath = os.path.join(output_psd_dir, vis_filename)
            print(f"  Saving PSD visualization to: {vis_filepath}")
            save_psd_visualization(avg_psd, vis_filepath, title=f"Average 2D PSD - {folder_label}")
    else:
        print(f"  No valid PSDs calculated for folder '{folder_label}' or no images processed. Skipping saving.")
        logging.warning(f"No PSDs to average for {folder_label}. Processed count: {count}.")

    print("\n----------------------------------")
    logging.info(f"PSD analysis for folder '{folder_label}' complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and save average Power Spectral Density (PSD) for images in a specified folder.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing the images to be analyzed. The folder's name will be used as the label.")
    parser.add_argument("--output_dir_psd", type=str, default=None,
                        help="Directory to save the calculated average PSD .npy files and visualizations. Defaults to '[image_dir]/psd_analysis'.")
    parser.add_argument("--save_visualizations", action='store_true',
                        help="If set, save PNG visualizations of the average PSDs (using log scale).")
    args = parser.parse_args()
    main(args)