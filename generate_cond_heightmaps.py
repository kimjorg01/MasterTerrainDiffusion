import os
import numpy as np
import argparse
import csv
from noise import pnoise2
from PIL import Image

def generate_heightmap(width, height, scale, octaves, persistence, lacunarity, seed_offset):
    """
    Generate a single heightmap using Perlin noise.
    The seed is incorporated by offsetting the noise coordinates.
    """
    heightmap = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            nx = x / width
            ny = y / height
            value = pnoise2(nx * scale + seed_offset,
                            ny * scale + seed_offset,
                            octaves=octaves,
                            persistence=persistence,
                            lacunarity=lacunarity,
                            repeatx=1024,
                            repeaty=1024,
                            base=0)
            heightmap[y][x] = value
    return heightmap

def normalize_heightmap(heightmap):
    """
    Normalize the heightmap values to the range 0-255 (8-bit grayscale).
    """
    min_val = np.min(heightmap)
    max_val = np.max(heightmap)
    norm = (heightmap - min_val) / (max_val - min_val)
    norm = (norm * 255).astype(np.uint8)
    return norm

def remap_flat_heightmap(heightmap, low=0.4, high=0.6, brightness_offset=30):
    """
    Remap a heightmap for flat terrain.
    Assumes raw noise values are in roughly [-0.5, 0.5].
    Scales and clips values to be within [low, high],
    scales to 0-255, then subtracts a brightness offset.
    """
    
    remapped = 0.5 + heightmap * (high - 0.5)
    remapped = np.clip(remapped, low, high)
    
    
    remapped = ((remapped - low) / (high - low)) * 255
    
    
    remapped = remapped - brightness_offset
    remapped = np.clip(remapped, 0, 255)
    
    return remapped.astype(np.uint8)



def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    annotation_file = os.path.join(args.output_dir, "annotations.csv")
    
    
    terrain_types = {
        "flat": {
            "scale": 1.5,    
            "octaves": 3,     
            "persistence": 0.4,
            "lacunarity": 2.5
        },
        "rolling": {
            "scale": 1.7,     
            "octaves": 4,     
            "persistence": 0.5,
            "lacunarity": 2.0
        },
        "mountainous": {
            "scale": 2.0,     
            "octaves": 4,     
            "persistence": 0.5,
            "lacunarity": 2.0
        },
        "canyon": {
            "scale": 1.5,     
            "octaves": 8,
            "persistence": 0.5,
            "lacunarity": 2.0
        }
    }
    
    annotations = []  

    for label, params in terrain_types.items():
        for i in range(args.images_per_type):
            seed_offset = np.random.randint(0, 10000)
            heightmap = generate_heightmap(
                width=args.width,
                height=args.height,
                scale=params["scale"],
                octaves=params["octaves"],
                persistence=params["persistence"],
                lacunarity=params["lacunarity"],
                seed_offset=seed_offset
            )
            if label == "flat":
                norm_heightmap = remap_flat_heightmap(heightmap, low=0.4, high=0.6, brightness_offset=20)
            else:
                norm_heightmap = normalize_heightmap(heightmap)

            
            img = Image.fromarray(norm_heightmap, mode='L')
            filename = f"{label}_{i:03d}.png"
            filepath = os.path.join(args.output_dir, filename)
            img.save(filepath)
            

            print(f"Saved {filepath}")
            
            annotations.append({
                "filename": filename,
                "label": label,
                "scale": params["scale"],
                "octaves": params["octaves"],
                "persistence": params["persistence"],
                "lacunarity": params["lacunarity"],
                "seed_offset": seed_offset
            })
    
    
    keys = annotations[0].keys()
    with open(annotation_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(annotations)
    
    print(f"Annotations saved to {annotation_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an annotated dataset of heightmaps with different terrain types using Perlin noise.")
    parser.add_argument("--width", type=int, default=256, help="Width of each heightmap image in pixels")
    parser.add_argument("--height", type=int, default=256, help="Height of each heightmap image in pixels")
    parser.add_argument("--images_per_type", type=int, default=10, help="Number of images to generate per terrain type")
    parser.add_argument("--output_dir", type=str, default="annotated_dataset", help="Output directory for generated images and annotations")
    
    args = parser.parse_args()
    main(args)
