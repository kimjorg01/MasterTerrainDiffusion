import os
import numpy as np
from skimage import io, metrics

def compute_metrics(real_dir, gen_dir):
    real_files = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.png')])
    gen_files  = sorted([os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith('.png')])
    
    ssim_scores = []
    rmse_scores = []
    
    for real_path, gen_path in zip(real_files, gen_files):
        real_img = io.imread(real_path, as_gray=True).astype(np.float32)
        gen_img  = io.imread(gen_path, as_gray=True).astype(np.float32)
        
        
        ssim = metrics.structural_similarity(real_img, gen_img, data_range=real_img.max() - real_img.min())
        rmse = np.sqrt(metrics.mean_squared_error(real_img, gen_img))
        
        ssim_scores.append(ssim)
        rmse_scores.append(rmse)
    
    print("Average SSIM:", np.mean(ssim_scores))
    print("Average RMSE:", np.mean(rmse_scores))


compute_metrics("dataset/heightmaps", "generated")
