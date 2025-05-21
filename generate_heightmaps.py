import os
import numpy as np
from noise import pnoise2
from PIL import Image
import argparse

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

def main(args):
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    for i in range(args.num_images):
        
        seed_offset = np.random.randint(0, 10000)
        heightmap = generate_heightmap(
            width=args.width,
            height=args.height,
            scale=args.scale,
            octaves=args.octaves,
            persistence=args.persistence,
            lacunarity=args.lacunarity,
            seed_offset=seed_offset
        )
        norm_heightmap = normalize_heightmap(heightmap)
        img = Image.fromarray(norm_heightmap, mode='L')
        
        
        filename = os.path.join(args.output_dir, f"heightmap_{i:03d}.png")
        img.save(filename)
        print(f"Saved {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a dataset of heightmaps using Perlin noise.")
    parser.add_argument("--num_images", type=int, default=500, help="Number of heightmap images to generate")
    parser.add_argument("--width", type=int, default=256, help="Width of each heightmap image in pixels")
    parser.add_argument("--height", type=int, default=256, help="Height of each heightmap image in pixels")
    parser.add_argument("--scale", type=float, default=2.0, help="Scale factor for noise frequency")
    parser.add_argument("--octaves", type=int, default=6, help="Number of octaves for noise generation")
    parser.add_argument("--persistence", type=float, default=0.5, help="Persistence for noise generation")
    parser.add_argument("--lacunarity", type=float, default=2.0, help="Lacunarity for noise generation")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output directory for generated images")
    
    args = parser.parse_args()
    main(args)










