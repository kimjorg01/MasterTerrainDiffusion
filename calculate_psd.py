import os
import numpy as np
import pandas as pd
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
        
        plt.imshow(psd_array, cmap='magma', norm=LogNorm(vmin=1e-2, vmax=psd_array.max())) 
        plt.colorbar(label='Log Power')
        plt.title(title)
        
        plt.xticks([])
        plt.yticks([])
        plt.savefig(output_path, bbox_inches='tight')
        plt.close() 
        logging.debug(f"Saved PSD visualization to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save PSD visualization {output_path}: {e}")


def main(args):
    annotations_path = os.path.join(args.data_dir, "annotations.csv")
    output_psd_dir = args.output_dir_psd or os.path.join(args.data_dir, "psd_analysis") 

    if not os.path.exists(annotations_path):
        logging.error(f"Annotations file not found: {annotations_path}")
        return

    os.makedirs(output_psd_dir, exist_ok=True)
    logging.info(f"PSD results will be saved in: {output_psd_dir}")

    logging.info(f"Loading annotations from: {annotations_path}")
    try:
        df = pd.read_csv(annotations_path)
        logging.info(f"Loaded {len(df)} records from annotations.")
    except Exception as e:
        logging.error(f"Error reading CSV {annotations_path}: {e}")
        return

    
    
    psd_accumulators = defaultdict(lambda: {"sum": None, "count": 0})

    logging.info("Calculating 2D PSD for dataset images...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images for PSD"):
        filename = row['filename']
        label = row['label']
        if not isinstance(label, str) or not label:
             logging.warning(f"Skipping row {index} due to invalid label: {label}")
             continue
        image_path = os.path.join(args.data_dir, filename)

        psd_2d = calculate_psd_2d(image_path)

        if psd_2d is not None:
            accumulator = psd_accumulators[label]
            if accumulator["sum"] is None:
                
                accumulator["sum"] = np.zeros_like(psd_2d, dtype=np.float64) 

            
            if accumulator["sum"].shape == psd_2d.shape:
                accumulator["sum"] += psd_2d
                accumulator["count"] += 1
            else:
                logging.warning(f"Shape mismatch for {filename} ({psd_2d.shape}) vs expected ({accumulator['sum'].shape}). Skipping.")


    
    logging.info("Calculating and saving average PSD per label...")
    print("\n--- Average PSD Calculation Summary ---")

    average_psds = {} 

    sorted_labels = sorted(psd_accumulators.keys())
    for label in sorted_labels:
        print(f"\nClass: {label}")
        accumulator = psd_accumulators[label]
        count = accumulator["count"]
        psd_sum = accumulator["sum"]

        if count > 0 and psd_sum is not None:
            avg_psd = psd_sum / count
            average_psds[label] = avg_psd
            print(f"  Calculated average PSD based on {count} images.")

            
            npy_filename = f"{label}_average_psd.npy"
            npy_filepath = os.path.join(output_psd_dir, npy_filename)
            try:
                np.save(npy_filepath, avg_psd)
                print(f"  Saved average PSD data to: {npy_filepath}")
            except Exception as e:
                logging.error(f"Failed to save average PSD .npy for {label}: {e}")
                print(f"  Error saving average PSD data for {label}.")

            
            if args.save_visualizations:
                vis_filename = f"{label}_average_psd.png"
                vis_filepath = os.path.join(output_psd_dir, vis_filename)
                print(f"  Saving PSD visualization to: {vis_filepath}")
                save_psd_visualization(avg_psd, vis_filepath, title=f"Average 2D PSD - {label}")

        else:
            print(f"  No valid PSDs calculated for class '{label}'. Skipping saving.")
            average_psds[label] = None

    print("\n----------------------------------")
    logging.info("PSD analysis complete.")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and save average Power Spectral Density (PSD) for image classes in a dataset.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the images and annotations.csv file.")
    parser.add_argument("--output_dir_psd", type=str, default=None,
                        help="Directory to save the calculated average PSD .npy files and visualizations. Defaults to '[data_dir]/psd_analysis'.")
    parser.add_argument("--save_visualizations", action='store_true',
                        help="If set, save PNG visualizations of the average PSDs (using log scale).")
    args = parser.parse_args()
    main(args)