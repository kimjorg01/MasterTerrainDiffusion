import os
import numpy as np
import argparse
import csv
from noise import pnoise2
from PIL import Image
from tqdm import tqdm

def generate_heightmap(width, height, scale, octaves, persistence, lacunarity, seed_offset, detail=0.0, invert_detail=False):
    """
    Generate a heightmap using Perlin noise, with an optional ridged noise detail.
    
    If invert_detail is False (default), the ridged noise is added to base noise (for mountain ridges).
    If invert_detail is True, the ridged noise is subtracted to create depressions (simulating rivers).
    """
    heightmap = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            nx = x / width
            ny = y / height
            
            
            base_value = pnoise2(nx * scale + seed_offset,
                                 ny * scale + seed_offset,
                                 octaves=octaves,
                                 persistence=persistence,
                                 lacunarity=lacunarity,
                                 repeatx=1024,
                                 repeaty=1024,
                                 base=0)
            
            ridge = 0.0
            amplitude = 1.0
            frequency = scale
            for o in range(octaves):
                n = pnoise2(nx * frequency + seed_offset, ny * frequency + seed_offset, base=0)
                n = 1.0 - abs(n)   
                n = n * n          
                ridge += n * amplitude
                amplitude *= persistence
                frequency *= lacunarity
            
            
            
            if invert_detail:
                combined = base_value - detail * ridge
            else:
                combined = base_value + detail * ridge
            
            heightmap[y][x] = combined
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
    Scales and clips values to be within [low, high],
    scales to 0-255, then subtracts a brightness offset.
    """
    remapped = 0.5 + heightmap * (high - 0.5)
    remapped = np.clip(remapped, low, high)
    remapped = ((remapped - low) / (high - low)) * 255
    remapped = remapped - brightness_offset
    remapped = np.clip(remapped, 0, 255)
    return remapped.astype(np.uint8)

def apply_radial_gradient(heightmap, exponent=2.0, weight=1.0):
    """
    Apply a radial gradient to the heightmap to simulate a volcanic effect.
    The gradient peaks at the center and falls off toward the edges.
    
    Args:
        heightmap (np.array): 2D heightmap array.
        exponent (float): Controls the steepness of the falloff (higher means steeper).
        weight (float): How strongly the gradient affects the heightmap.
    
    Returns:
        Modified heightmap with the radial gradient applied.
    """
    h, w = heightmap.shape
    cy, cx = h / 2, w / 2
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_distance = np.sqrt(cx**2 + cy**2)
    
    mask = 1.0 - (distance / max_distance) ** exponent
    
    return heightmap + weight * mask

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    annotation_file = os.path.join(args.output_dir, "annotations.csv")

    
    terrain_types = {
        "flat": {
            "scale": 1.5, "octaves": 8, "persistence": 0.5, "lacunarity": 2.0,
            "detail": 0.0, "invert_detail": False, "apply_radial": False
        },
        "mountain_ridges": {
            "scale": 1.5, "octaves": 8, "persistence": 0.5, "lacunarity": 2.0,
            "detail": 1.0, "invert_detail": False, "apply_radial": False
        },
        "mountain_rivers": {
            "scale": 1.5, "octaves": 8, "persistence": 0.5, "lacunarity": 2.0,
            "detail": 0.6, "invert_detail": True, "apply_radial": False
        }
    }

    annotations = [] 
    total_images_generated = 0

    
    for label, params in terrain_types.items():
        print(f"\nGenerating {args.images_per_type} images for type: '{label}'...") 

        
        
        for i in tqdm(range(args.images_per_type), desc=f"{label:>15}"): 
            
            seed_offset = np.random.randint(0, 10000) 

            
            heightmap = generate_heightmap(
                width=args.width,
                height=args.height,
                scale=params["scale"],
                octaves=params["octaves"],
                persistence=params["persistence"],
                lacunarity=params["lacunarity"],
                seed_offset=seed_offset,
                detail=params["detail"],
                invert_detail=params["invert_detail"]
            )

            
            if params.get("apply_radial", False):
                exp = params.get("radial_exponent", 2.0)
                weight = params.get("radial_weight", 1.0)
                heightmap = apply_radial_gradient(heightmap, exponent=exp, weight=weight)

            
            
            norm_heightmap = normalize_heightmap(heightmap)

            
            img = Image.fromarray(norm_heightmap, mode='L')

            
            filename = f"{label}_{i:04d}.png" 
            filepath = os.path.join(args.output_dir, filename)

            
            try:
                img.save(filepath)
            except Exception as e:
                print(f"\nError saving image {filepath}: {e}") 
                continue 

            
            annotations.append({
                "filename": filename,
                "label": label,
                "scale": params["scale"],
                "octaves": params["octaves"],
                "persistence": params["persistence"],
                "lacunarity": params["lacunarity"],
                "detail": params["detail"],
                "invert_detail": params["invert_detail"],
                "apply_radial": params.get("apply_radial", False),
                "seed_offset": seed_offset
            })
            total_images_generated += 1

    
    if not annotations:
        print("\nNo images were generated or saved successfully. Skipping annotation file.")
        return

    print(f"\nGenerated {total_images_generated} images in total.")
    print(f"Writing annotations to {annotation_file}...")

    try:
        keys = annotations[0].keys()
        with open(annotation_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(annotations)
        print("Annotations saved successfully.")
    except Exception as e:
        print(f"\nError writing annotations file {annotation_file}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an annotated dataset of heightmaps...") 
    parser.add_argument("--width", type=int, default=256, help="Width of each heightmap image")
    parser.add_argument("--height", type=int, default=256, help="Height of each heightmap image")
    
    parser.add_argument("--images_per_type", type=int, default=50, help="Number of images per terrain type")
    parser.add_argument("--output_dir", type=str, default="ClassifierTests", help="Output directory")

    args = parser.parse_args()

    
    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 1: 
        print(f"Warning: Output directory '{args.output_dir}' already exists and may contain files.")
        
        
        
        
        

    main(args)
    print("\nDataset generation complete.")
