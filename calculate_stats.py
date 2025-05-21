import os
import numpy as np
import pandas as pd
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
    annotations_path = os.path.join(args.data_dir, "annotations.csv")
    if not os.path.exists(annotations_path):
        logging.error(f"Annotations file not found: {annotations_path}")
        return

    logging.info(f"Loading annotations from: {annotations_path}")
    try:
        df = pd.read_csv(annotations_path)
        logging.info(f"Loaded {len(df)} records from annotations.")
    except Exception as e:
        logging.error(f"Error reading CSV {annotations_path}: {e}")
        return

    
    
    label_stats = defaultdict(lambda: defaultdict(list))

    logging.info("Calculating statistics for dataset images...")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
        filename = row['filename']
        label = row['label']
        
        if not isinstance(label, str) or not label:
             logging.warning(f"Skipping row {index} due to invalid label: {label}")
             continue
        image_path = os.path.join(args.data_dir, filename)

        stats = calculate_image_stats(image_path)

        if stats:
            label_stats[label]["mean_height"].append(stats["mean_height"])
            label_stats[label]["std_dev"].append(stats["std_dev"])
            label_stats[label]["avg_grad_mag"].append(stats["avg_grad_mag"])

    
    logging.info("Calculating average statistics per label...")
    print("\n--- Dataset Statistics Summary ---")

    sorted_labels = sorted(label_stats.keys()) 

    
    average_stats_summary = {}

    for label in sorted_labels:
        print(f"\nClass: {label}")
        stats_for_label = label_stats[label]
        
        mean_height_list = stats_for_label.get("mean_height", [])
        std_dev_list = stats_for_label.get("std_dev", [])
        grad_mag_list = stats_for_label.get("avg_grad_mag", [])

        num_images = len(mean_height_list) 
        print(f"  (Based on {num_images} successfully processed images)")

        
        average_stats_summary[label] = {}

        if num_images > 0:
            mean_h_avg = np.mean(mean_height_list)
            mean_h_std = np.std(mean_height_list) 
            average_stats_summary[label]["mean_height_avg"] = mean_h_avg
            average_stats_summary[label]["mean_height_std_across_images"] = mean_h_std
            print(f"  Mean Height:          Avg={mean_h_avg:.4f} (Std Dev across images={mean_h_std:.4f})")

            std_dev_avg = np.mean(std_dev_list)
            std_dev_std = np.std(std_dev_list) 
            average_stats_summary[label]["std_dev_avg"] = std_dev_avg
            average_stats_summary[label]["std_dev_std_across_images"] = std_dev_std
            print(f"  Std Dev (Roughness):  Avg={std_dev_avg:.4f} (Std Dev across images={std_dev_std:.4f})")

            grad_mag_avg = np.mean(grad_mag_list)
            grad_mag_std = np.std(grad_mag_list) 
            average_stats_summary[label]["avg_grad_mag_avg"] = grad_mag_avg
            average_stats_summary[label]["avg_grad_mag_std_across_images"] = grad_mag_std
            print(f"  Avg Grad Mag (Steepness): Avg={grad_mag_avg:.4f} (Std Dev across images={grad_mag_std:.4f})")
        else:
            print("  No statistics calculated for this class (likely image processing errors).")
            
            average_stats_summary[label] = {
                "mean_height_avg": None, "mean_height_std_across_images": None,
                "std_dev_avg": None, "std_dev_std_across_images": None,
                "avg_grad_mag_avg": None, "avg_grad_mag_std_across_images": None,
                "num_images": 0
            }
        average_stats_summary[label]["num_images_processed"] = num_images


    print("\n----------------------------------")

    
    output_file = os.path.join(args.data_dir, "dataset_stats_summary.json")
    logging.info(f"Attempting to save average statistics summary to {output_file}")
    try:
        with open(output_file, 'w') as f:
            
            json.dump(average_stats_summary, f, indent=4)
        print(f"Average statistics summary saved successfully to {output_file}")
        logging.info(f"Average statistics saved to {output_file}")
    except TypeError as e:
         logging.error(f"Failed to save statistics summary due to non-serializable data: {e}")
         print(f"Error: Could not save statistics to JSON. Check for non-serializable types like numpy floats.")
         
         
         
         
         
         
         
         
         

    except Exception as e:
        logging.error(f"Failed to save statistics summary: {e}")
        print(f"Error saving statistics summary: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate feature statistics for images in an annotated dataset.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the images and annotations.csv file.")
    args = parser.parse_args()
    main(args)