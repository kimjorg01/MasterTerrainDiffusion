import os
import math
import csv
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




class HeightmapDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform=None):
        """
        Expects an annotations CSV with 'filename' and 'label' columns.
        """
        self.annotations = pd.read_csv(annotations_file)
        self.images_dir = images_dir
        self.transform = transform
        
        self.label_to_idx = {"flat": 0, "mountain_ridges": 1, "mountain_rivers": 2}
        logging.info(f"Loaded dataset with {len(self.annotations)} images.")
        logging.info(f"Label mapping: {self.label_to_idx}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.images_dir, row['filename'])
        try:
            
            img = Image.open(img_path).convert("L")
        except FileNotFoundError:
            logging.error(f"Image file not found: {img_path}")
            
            
            
            return None, None 
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            return None, None

        label = row['label']
        if label not in self.label_to_idx:
             logging.warning(f"Label '{label}' in {row['filename']} not in predefined mapping. Skipping?")
             
             label_idx = 0
        else:
             label_idx = self.label_to_idx[label]

        if self.transform:
            img = self.transform(img)

        
        condition = torch.zeros(len(self.label_to_idx))
        condition[label_idx] = 1.0
        return img, condition


def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.empty(0), torch.empty(0) 
    return torch.utils.data.dataloader.default_collate(batch)





def get_transform(args):
    
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor(),                  
        
        transforms.Lambda(scale_to_neg_one_to_one)
    ])




def get_timestep_embedding(timesteps, embedding_dim):
    assert embedding_dim % 2 == 0, "Embedding dimension must be even"
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    
    
    return emb




class ConditionalUNet(nn.Module):
    def __init__(self, time_emb_dim=128, cond_emb_dim=128, num_labels=3, base_channels=64, num_groups=8):
        super(ConditionalUNet, self).__init__()

        
        if time_emb_dim % 2 != 0:
            raise ValueError("time_emb_dim must be even")

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(), 
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(num_labels, cond_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_emb_dim * 4, cond_emb_dim)
        )

        
        
        self.cond_bottleneck_proj = nn.Linear(time_emb_dim, base_channels * 4)
        

        
        def conv_block(in_c, out_c, kernel_size=3, stride=1, padding=1):
            
            current_num_groups = min(num_groups, out_c) if out_c >= num_groups else 1 
            if out_c % current_num_groups != 0: 
                 current_num_groups = 1 if out_c > 0 else num_groups 
                 if out_c > 0 : logging.warning(f"GroupNorm: out_c {out_c} not divisible by num_groups {num_groups}. Using 1 group.")


            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.GroupNorm(current_num_groups, out_c),
                nn.SiLU()
            )

        
        self.enc1_1 = conv_block(1, base_channels)
        self.enc1_2 = conv_block(base_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1) 

        self.enc2_1 = conv_block(base_channels, base_channels * 2)
        self.enc2_2 = conv_block(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1) 

        
        self.bottle1 = conv_block(base_channels * 2, base_channels * 4)
        self.bottle2 = conv_block(base_channels * 4, base_channels * 4)

        
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1) 
        self.dec1_1 = conv_block(base_channels * 4, base_channels * 2) 
        self.dec1_2 = conv_block(base_channels * 2, base_channels * 2)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1) 
        self.dec2_1 = conv_block(base_channels * 2, base_channels) 
        self.dec2_2 = conv_block(base_channels, base_channels)

        
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)


    def forward(self, x, t, cond):
        
        t_emb = get_timestep_embedding(t, 128) 
        t_emb = self.time_mlp(t_emb)
        c_emb = self.cond_mlp(cond)
        cond_emb = t_emb + c_emb 

        
        cond_emb_projected = self.cond_bottleneck_proj(cond_emb) 
        

        
        x1 = self.enc1_1(x)
        x1 = self.enc1_2(x1) 

        x2 = self.down1(x1)
        x2 = self.enc2_1(x2)
        x2 = self.enc2_2(x2) 

        x3 = self.down2(x2)

        
        x_bottle = self.bottle1(x3)
        x_bottle = self.bottle2(x_bottle)

        
        
        cond_emb_sp = cond_emb_projected.unsqueeze(-1).unsqueeze(-1) 
        
        x_bottle = x_bottle + cond_emb_sp 
        


        
        x4 = self.up1(x_bottle)
        x4 = torch.cat([x4, x2], dim=1) 
        x4 = self.dec1_1(x4)
        x4 = self.dec1_2(x4)

        x5 = self.up2(x4)
        x5 = torch.cat([x5, x1], dim=1) 
        x5 = self.dec2_1(x5)
        x5 = self.dec2_2(x5)

        out = self.out_conv(x5)
        return out





def create_diffusion_schedule(T, beta_start=1e-4, beta_end=0.02, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def scale_to_neg_one_to_one(t):
    """Scales a tensor from [0, 1] to [-1, 1]"""
    return t * 2.0 - 1.0




@torch.no_grad() 
def sample_images(model, condition, T, shape, betas, alphas_cumprod, device):
    model.eval() 

    x = torch.randn(shape, device=device) 

    alphas = 1.0 - betas
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
    posterior_variance = betas * (1.0 - torch.cat([alphas_cumprod[0:1], alphas_cumprod[:-1]])) / (1.0 - alphas_cumprod) 

    num_images = shape[0]

    for t in reversed(range(T)):
        t_tensor = torch.full((num_images,), t, device=device, dtype=torch.long)

        
        noise_pred = model(x, t_tensor, condition)

        
        beta_t = betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = sqrt_recip_alphas[t].view(-1, 1, 1, 1)

        mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred)

        if t == 0:
            x = mean 
        else:
            
            variance = posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            
            x = mean + torch.sqrt(variance) * noise

    model.train() 
    return x

def tensor_to_image(tensor):
    tensor = (tensor + 1) / 2  
    tensor = tensor.clamp(0, 1)
    
    array = tensor.detach().cpu().numpy().squeeze() 
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")




def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    
    transform = get_transform(args)
    annotations_file = os.path.join(args.data_dir, "annotations.csv")
    dataset = HeightmapDataset(annotations_file, args.data_dir, transform=transform)

    
    if len(dataset) == 0:
         logging.error("Dataset is empty. Please check data path and annotations file.")
         return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True, 
        collate_fn=collate_fn 
    )
    logging.info(f"DataLoader created with batch size {args.batch_size} and {args.num_workers} workers.")

    
    
    num_labels = len(dataset.label_to_idx)
    model = ConditionalUNet(
        time_emb_dim=args.time_emb_dim,
        cond_emb_dim=args.cond_emb_dim,
        num_labels=num_labels,
        base_channels=args.base_channels,
        num_groups=args.num_groups
    ).to(device)
    logging.info(f"Model ConditionalUNet created with {sum(p.numel() for p in model.parameters())} parameters.")

    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) 
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader), eta_min=1e-6) 
    logging.info(f"Optimizer AdamW and CosineAnnealingLR scheduler created.")

    
    
    criterion = nn.HuberLoss(delta=0.1)
    

    
    T = args.T
    betas, alphas_schedule, alphas_cumprod = create_diffusion_schedule(T, device=device)
    logging.info(f"Diffusion schedule created with T={T} steps.")

    
    start_epoch = 0
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            logging.info(f"Loading checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            logging.info(f"Resuming training from epoch {start_epoch + 1}")
        else:
            logging.warning(f"Checkpoint file not found at {args.resume_from_checkpoint}. Starting from scratch.")

    
    label_map = {v: k for k, v in dataset.label_to_idx.items()} 

    
    logging.info(f"Starting training from epoch {start_epoch + 1} to {args.epochs}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for i, (images, cond) in enumerate(progress_bar):
             
            if images.numel() == 0:
                 logging.warning(f"Skipping empty batch at step {i} in epoch {epoch+1}.")
                 continue

            images = images.to(device)
            cond = cond.to(device)
            B = images.shape[0]

            
            t = torch.randint(0, T, (B,), device=device).long()

            
            noise = torch.randn_like(images).to(device)
            alpha_t_cumprod = alphas_cumprod[t].view(-1, 1, 1, 1)
            noisy_images = torch.sqrt(alpha_t_cumprod) * images + torch.sqrt(1.0 - alpha_t_cumprod) * noise

            
            optimizer.zero_grad()
            noise_pred = model(noisy_images, t, cond)
            loss = criterion(noise_pred, noise)

            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            
            scheduler.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        epoch_loss = running_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{args.epochs} Average Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        
        if (epoch + 1) % args.save_checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1:04d}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'args': args 
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

            
            model.eval() 
            sample_epoch_dir = os.path.join(args.sample_dir, f"epoch_{epoch+1:04d}")
            os.makedirs(sample_epoch_dir, exist_ok=True)
            logging.info(f"Generating samples for epoch {epoch+1} into {sample_epoch_dir}")

            num_cond = len(label_map)
            shape = (args.samples_per_condition_checkpoint, 1, args.image_size, args.image_size)

            for label_idx, label_name in label_map.items():
                condition = torch.zeros((args.samples_per_condition_checkpoint, num_cond), device=device)
                condition[:, label_idx] = 1.0

                samples = sample_images(model, condition, T, shape, betas, alphas_cumprod, device)

                for i in range(args.samples_per_condition_checkpoint):
                    sample_img = tensor_to_image(samples[i:i+1]) 
                    out_path = os.path.join(sample_epoch_dir, f"{label_name}_sample_{i:02d}.png")
                    try:
                        sample_img.save(out_path)
                    except Exception as e:
                        logging.error(f"Failed to save sample image {out_path}: {e}")

            logging.info(f"Finished generating samples for epoch {epoch+1}")
            model.train() 

    logging.info("Training finished.")

    
    final_model_path = os.path.join(args.save_dir, "conditional_diffusion_final.pth")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model state dict saved to {final_model_path}")





def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device} for sampling.")

    
    label_to_idx = {"flat": 0, "mountain_ridges": 1, "mountain_rivers": 2}
    num_labels = len(label_to_idx)

    if args.condition not in label_to_idx:
        raise ValueError(f"Invalid condition '{args.condition}'. Must be one of {list(label_to_idx.keys())}")

    cond_idx = label_to_idx[args.condition]
    condition = torch.zeros((args.num_samples, num_labels), device=device)
    condition[:, cond_idx] = 1.0

    
    
    model = ConditionalUNet(
        time_emb_dim=args.time_emb_dim,
        cond_emb_dim=args.cond_emb_dim,
        num_labels=num_labels,
        base_channels=args.base_channels,
        num_groups=args.num_groups
    ).to(device)

    logging.info(f"Loading model state dict from: {args.model_path}")
    
    loaded_data = torch.load(args.model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in loaded_data:
        model.load_state_dict(loaded_data['model_state_dict'])
    else:
        model.load_state_dict(loaded_data)

    model.eval() 

    
    T = args.T 
    betas, _, alphas_cumprod = create_diffusion_schedule(T, device=device)
    logging.info(f"Diffusion schedule created for sampling with T={T} steps.")

    
    shape = (args.num_samples, 1, args.image_size, args.image_size)
    logging.info(f"Generating {args.num_samples} samples for condition '{args.condition}'...")
    samples = sample_images(model, condition, T, shape, betas, alphas_cumprod, device)
    logging.info("Sample generation complete.")

    
    output_sample_dir = "OutputDir" 
    os.makedirs(output_sample_dir, exist_ok=True)
    for i in range(args.num_samples):
        sample_img = tensor_to_image(samples[i:i+1])
        out_path = os.path.join(output_sample_dir, f"{args.condition}_sample_{i:03d}.png")
        try:
            sample_img.save(out_path)
            logging.info(f"Saved sample to {out_path}")
        except Exception as e:
            logging.error(f"Failed to save sample image {out_path}: {e}")




def main_mode():
    parser = argparse.ArgumentParser(description="Conditional Diffusion Model: Train or Sample for Heightmap Generation.")

    
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True, help="Mode: 'train' or 'sample'.")
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion steps.")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the images (height and width). Assumed square.")
    parser.add_argument("--time_emb_dim", type=int, default=128, help="Dimension for time embeddings (must be even).")
    parser.add_argument("--cond_emb_dim", type=int, default=128, help="Dimension for condition embeddings.")
    parser.add_argument("--base_channels", type=int, default=64, help="Base number of channels in U-Net.")
    parser.add_argument("--num_groups", type=int, default=8, help="Number of groups for Group Normalization in U-Net.")

    
    parser.add_argument("--data_dir", type=str, default="annotated_dataset3", help="Directory containing images and annotations.csv (train mode).")
    parser.add_argument("--save_dir", type=str, default="models_output", help="Directory to save checkpoints and final model (train mode).")
    parser.add_argument("--sample_dir", type=str, default="training_samples", help="Directory to save samples generated during training checks (train mode).")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (train mode). Adjust based on dataset size.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (train mode). Adjust based on VRAM.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (train mode).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader (train mode). Adjust based on CPU.")
    parser.add_argument("--save_checkpoint_freq", type=int, default=25, help="Save checkpoint every N epochs (train mode).")
    parser.add_argument("--samples_per_condition_checkpoint", type=int, default=2, help="Number of samples per condition to generate at each checkpoint (train mode).")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a .pth checkpoint file to resume training from (train mode).")

    
    parser.add_argument("--model_path", type=str, default="models_output/conditional_diffusion_final.pth", help="Path to the trained model state dict or checkpoint (.pth) for sampling (sample mode).")
    parser.add_argument("--condition", type=str, choices=["flat", "mountain_ridges", "mountain_rivers"], default="flat", help="Terrain type condition to sample (sample mode).")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate (sample mode).")

    args = parser.parse_args()



    if args.mode == "train":
        logging.info("Selected mode: train")
        logging.info(f"Training Arguments: {vars(args)}")
        train(args)
    elif args.mode == "sample":
        logging.info("Selected mode: sample")
        logging.info(f"Sampling Arguments: {vars(args)}")
        sample(args)

if __name__ == "__main__":
    main_mode()

