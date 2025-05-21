import os
import numpy as np

from PIL import Image
import argparse
from tqdm import tqdm
from collections import defaultdict
import logging
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_image_stats(image_path):
    """
    Loads a grayscale image and calculates basic statistics.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: Dictionary containing 'mean_height', 'std_dev', 'avg_grad_mag',
              or None if the image cannot be processed.
    """
    try:
        img = Image.open(image_path).convert('L') 
        img_array = np.array(img, dtype=np.float32) / 255.0 

        
        mean_height = np.mean(img_array)
        std_dev = np.std(img_array)

        
        gy, gx = np.gradient(img_array)
        grad_mag = np.sqrt(gx**2 + gy**2)
        avg_grad_mag = np.mean(grad_mag)

        return {
            "mean_height": mean_height,
            "std_dev": std_dev,
            "avg_grad_mag": avg_grad_mag
        }

    except FileNotFoundError:
        logging.warning(f"File not found: {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None



def main(args):
    image_folder_path = args.image_dir 
    
    if not os.path.isdir(image_folder_path):
        logging.error(f"Provided path is not a directory: {image_folder_path}")
        print(f"Error: The path '{image_folder_path}' is not a valid directory.")
        return

    
    label_for_folder = os.path.basename(os.path.normpath(image_folder_path)) 

    logging.info(f"Processing images from folder: {image_folder_path} (Label: {label_for_folder})")

    
    
    label_stats = defaultdict(lambda: defaultdict(list))
    
    image_files = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif') 
    
    logging.info(f"Scanning for images in {image_folder_path}...")
    for f_name in os.listdir(image_folder_path):
        if f_name.lower().endswith(supported_extensions):
            image_files.append(os.path.join(image_folder_path, f_name))

    if not image_files:
        logging.warning(f"No image files found with supported extensions in {image_folder_path}")
        print(f"No images found in '{image_folder_path}'. Ensure images have extensions like {supported_extensions}.")
        
        
    else:
        logging.info(f"Found {len(image_files)} images to process in {image_folder_path}.")

    
    for image_path in tqdm(image_files, desc=f"Processing Images in '{label_for_folder}'"):
        stats = calculate_image_stats(image_path)
        if stats:
            label_stats[label_for_folder]["mean_height"].append(stats["mean_height"])
            label_stats[label_for_folder]["std_dev"].append(stats["std_dev"])
            label_stats[label_for_folder]["avg_grad_mag"].append(stats["avg_grad_mag"])

    
    logging.info(f"Calculating average statistics for label: {label_for_folder}...")
    print("\n--- Folder Statistics Summary ---")

    average_stats_summary = {}

    
    if label_for_folder in label_stats and label_stats[label_for_folder]["mean_height"]: 
        print(f"\nClass (from folder): {label_for_folder}")
        stats_for_label = label_stats[label_for_folder]
        
        mean_height_list = stats_for_label.get("mean_height", [])
        std_dev_list = stats_for_label.get("std_dev", [])
        grad_mag_list = stats_for_label.get("avg_grad_mag", [])

        num_images = len(mean_height_list) 
        print(f"  (Based on {num_images} successfully processed images)")

        average_stats_summary[label_for_folder] = {}

        if num_images > 0:
            mean_h_avg = np.mean(mean_height_list)
            mean_h_std = np.std(mean_height_list) 
            average_stats_summary[label_for_folder]["mean_height_avg"] = float(mean_h_avg)
            average_stats_summary[label_for_folder]["mean_height_std_across_images"] = float(mean_h_std)
            print(f"  Mean Height:              Avg={mean_h_avg:.4f} (Std Dev across images={mean_h_std:.4f})")

            std_dev_avg = np.mean(std_dev_list)
            std_dev_std = np.std(std_dev_list) 
            average_stats_summary[label_for_folder]["std_dev_avg"] = float(std_dev_avg)
            average_stats_summary[label_for_folder]["std_dev_std_across_images"] = float(std_dev_std)
            print(f"  Std Dev (Roughness):      Avg={std_dev_avg:.4f} (Std Dev across images={std_dev_std:.4f})")

            grad_mag_avg = np.mean(grad_mag_list)
            grad_mag_std = np.std(grad_mag_list)
            average_stats_summary[label_for_folder]["avg_grad_mag_avg"] = float(grad_mag_avg)
            average_stats_summary[label_for_folder]["avg_grad_mag_std_across_images"] = float(grad_mag_std)
            print(f"  Avg Grad Mag (Steepness): Avg={grad_mag_avg:.4f} (Std Dev across images={grad_mag_std:.4f})")
        
        

        average_stats_summary[label_for_folder]["num_images_processed"] = num_images

    else: 
        message = (f"No images were successfully processed in folder '{label_for_folder}'. "
                   f"Total images found: {len(image_files)}.")
        if not image_files:
             message = f"No image files with supported extensions found in folder '{label_for_folder}'."

        logging.warning(message)
        print(f"\nClass (from folder): {label_for_folder}")
        print(f"  {message}")
        average_stats_summary[label_for_folder] = {
            "message": message,
            "mean_height_avg": None, "mean_height_std_across_images": None,
            "std_dev_avg": None, "std_dev_std_across_images": None,
            "avg_grad_mag_avg": None, "avg_grad_mag_std_across_images": None,
            "num_images_processed": 0
        }

    print("\n----------------------------------")

    
    
    output_filename = f"{label_for_folder}_stats_summary.json"
    output_file = os.path.join(image_folder_path, output_filename)
    
    logging.info(f"Attempting to save average statistics summary to {output_file}")
    try:
        with open(output_file, 'w') as f:
            json.dump(average_stats_summary, f, indent=4)
        print(f"Average statistics summary saved successfully to {output_file}")
        logging.info(f"Average statistics saved to {output_file}")
    except TypeError as e: 
        logging.error(f"Failed to save statistics summary due to non-serializable data: {e}")
        print(f"Error: Could not save statistics to JSON. Check for non-serializable types. ({e})")
    except Exception as e:
        logging.error(f"Failed to save statistics summary: {e}")
        print(f"Error saving statistics summary: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate feature statistics for all images in a specified folder.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing the images to be analyzed. The folder's name will be used as the label.")
    args = parser.parse_args()
    main(args)