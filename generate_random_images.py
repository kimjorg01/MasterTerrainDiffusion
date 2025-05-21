import numpy as np
from PIL import Image
import os
import argparse
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_random_noise_images(base_output_dir="random_noise_dataset",
                                 num_sets=3,
                                 images_per_set=50,
                                 img_width=256,
                                 img_height=256,
                                 set_names=None):
    """
    Generates sets of random noise grayscale images.

    Args:
        base_output_dir (str): The base directory to save the generated image sets.
        num_sets (int): Number of different sets/classes of random images to generate.
        images_per_set (int): Number of random images to generate for each set.
        img_width (int): Width of the images.
        img_height (int): Height of the images.
        set_names (list, optional): List of names for the sets/subdirectories. 
                                     If None, default names like "Random_Set_1" will be used.
    """

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        logging.info(f"Created base output directory: {base_output_dir}")

    if set_names and len(set_names) != num_sets:
        logging.error("Number of provided set names does not match num_sets. Using default names.")
        set_names = None

    if not set_names:
        set_names = [f"Random_Set_{i+1}" for i in range(num_sets)]

    total_images_generated = 0

    for i in range(num_sets):
        set_name = set_names[i]
        set_dir = os.path.join(base_output_dir, set_name)

        if not os.path.exists(set_dir):
            os.makedirs(set_dir)
            logging.info(f"Created directory for set '{set_name}': {set_dir}")

        logging.info(f"Generating {images_per_set} images for set '{set_name}'...")
        for j in range(images_per_set):
            
            
            random_pixels = (np.random.rand(img_height, img_width) * 255).astype(np.uint8)

            
            img = Image.fromarray(random_pixels, mode='L')  

            
            img_filename = f"{set_name.lower()}_noise_{j:04d}.png"
            img_path = os.path.join(set_dir, img_filename)
            try:
                img.save(img_path)
                total_images_generated += 1
            except Exception as e:
                logging.error(f"Could not save image {img_path}: {e}")

        logging.info(f"Finished generating images for set '{set_name}'.")

    logging.info(f"--- Generation Complete ---")
    logging.info(f"Total random noise images generated: {total_images_generated}")
    logging.info(f"Images saved in subdirectories under: {os.path.abspath(base_output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sets of random noise grayscale images.")
    parser.add_argument("--output_dir", type=str, default="random_noise_dataset",
                        help="Base directory to save the generated image sets.")
    parser.add_argument("--num_sets", type=int, default=3,
                        help="Number of different sets/classes of random images to generate (e.g., to mimic your Flat, Ridges, Rivers structure).")
    parser.add_argument("--images_per_set", type=int, default=50,
                        help="Number of random images to generate for each set.")
    parser.add_argument("--width", type=int, default=256, help="Width of the images.")
    parser.add_argument("--height", type=int, default=256, help="Height of the images.")
    parser.add_argument("--set_names", type=str, nargs='+', default=None, 
                        help="Optional: List of names for the sets/subdirectories (e.g., Random_Flat Random_Ridges Random_Rivers). If not provided, defaults like Random_Set_1 will be used.")

    args = parser.parse_args()

    generate_random_noise_images(base_output_dir=args.output_dir,
                                 num_sets=args.num_sets,
                                 images_per_set=args.images_per_set,
                                 img_width=args.width,
                                 img_height=args.height,
                                 set_names=args.set_names)
