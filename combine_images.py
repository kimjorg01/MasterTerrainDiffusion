import math
import numpy as np
from PIL import Image

def smooth_cos_alpha(x, blend_size):
    """Cosine-based easing from 0 to 1 across blend_size."""
    return 0.5 - 0.5 * math.cos(math.pi * x / blend_size)

def blend_horizontal(left, right, blend_size):
    h = left.shape[0]
    
    blended = np.zeros((h, 256 + 256 - blend_size), dtype=np.float32)

    
    blended[:, :256 - blend_size] = left[:, :256 - blend_size]

    
    blended[:, (256 - blend_size + blend_size):] = right[:, blend_size:]

    
    for i in range(blend_size):
        alpha = smooth_cos_alpha(i, blend_size)
        l = left[:, 256 - blend_size + i]
        r = right[:, i]
        blended[:, 256 - blend_size + i] = (1 - alpha) * l + alpha * r

    return blended

def blend_vertical(top, bottom, blend_size):
    w = top.shape[1]
    blended = np.zeros((256 + 256 - blend_size, w), dtype=np.float32)

    
    blended[:256 - blend_size, :] = top[:256 - blend_size, :]

    
    blended[(256 - blend_size + blend_size):, :] = bottom[blend_size:, :]

    
    for i in range(blend_size):
        alpha = smooth_cos_alpha(i, blend_size)
        t = top[256 - blend_size + i, :]
        b = bottom[i, :]
        blended[256 - blend_size + i, :] = (1 - alpha) * t + alpha * b

    return blended

def load_grayscale(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)

def save_grayscale(arr, path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

def main():
    
    tl = load_grayscale("TempUnityRaw/flat_002.png")
    tr = load_grayscale("TempUnityRaw/mountain_ridges_001.png")
    bl = load_grayscale("TempUnityRaw/mountain_ridges_002.png")
    br = load_grayscale("TempUnityRaw/mountain_rivers_009.png")

    blend_size = 64

    
    top = blend_horizontal(tl, tr, blend_size)
    bottom = blend_horizontal(bl, br, blend_size)

    
    final = blend_vertical(top, bottom, blend_size)

    
    
    
    final_img = Image.fromarray(np.clip(final, 0, 255).astype(np.uint8), mode="L")
    final_upscaled = final_img.resize((512, 512), resample=Image.LANCZOS)
    
    
    final_upscaled.save("blended_upscaled2.png")
    print("✅ Blended image saved as blended_upscaled2.png")

if __name__ == "__main__":
    main()
