import argparse
import os
import numpy as np
from PIL import Image

def png_to_raw(input_path, output_path, normalize=False):
    """
    Convert an 8-bit grayscale PNG to a 16-bit RAW file for Unity.
    
    Args:
        input_path  (str): Path to the input .png file.
        output_path (str): Path to the output .raw file.
        normalize   (bool): If True, perform a min-max normalization before converting to 16-bit.
    """
    
    img = Image.open(input_path).convert('L')
    arr = np.array(img, dtype=np.float32)  

    height, width = arr.shape
    print(f"Image size: {width} x {height}")

    
    if normalize:
        min_val = arr.min()
        max_val = arr.max()
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val)
        else:
            arr = arr * 0.0
    else:
        arr /= 255.0

    arr = arr * 65535.0
    arr_16 = arr.astype(np.uint16)

    with open(output_path, 'wb') as f:
        f.write(arr_16.tobytes())

    print(f"✅ Converted {input_path} -> {output_path}")

    
    meta_path = os.path.splitext(output_path)[0] + "_resolution.txt"
    with open(meta_path, 'w') as f:
        f.write(f"{width} {height}\n")
    print(f"ℹ️  Saved resolution info to {meta_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert PNG heightmap to 16-bit RAW for Unity.")
    parser.add_argument("input", type=str, help="Input PNG file")
    parser.add_argument("output", type=str, help="Output RAW file")
    parser.add_argument("--normalize", action="store_true",
                        help="Perform min-max normalization before converting to 16-bit.")
    args = parser.parse_args()

    png_to_raw(args.input, args.output, normalize=args.normalize)

if __name__ == "__main__":
    main()
