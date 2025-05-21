import os
import math
import csv
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class HeightmapDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform=None):
        """
        Expects an annotations CSV with a 'filename' column and a 'label' column.
        """
        self.annotations = pd.read_csv(annotations_file)
        self.images_dir = images_dir
        self.transform = transform
        self.label_to_idx = {"flat": 0, "mountain_ridges": 1, "mountain_rivers": 2}
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.images_dir, row['filename'])
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        
        label = row['label']
        label_idx = self.label_to_idx[label]
        condition = torch.zeros(3)
        condition[label_idx] = 1.0
        return img, condition


transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Lambda(lambda t: t * 2 - 1)  
])


def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class ConditionalUNet(nn.Module):
    def __init__(self, time_emb_dim=128, cond_emb_dim=128, base_channels=64):
        super(ConditionalUNet, self).__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_emb_dim),
            nn.ReLU()
        )
        
        
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x, t, cond):
        
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)  
        c_emb = self.cond_mlp(cond)   
        cond_emb = t_emb + c_emb      
        cond_emb = cond_emb[:, :, None, None]  
        
        x1 = self.enc1(x)             
        x2 = self.enc2(x1)            
        x3 = self.bottleneck(x2) + cond_emb  
        x4 = self.dec1(x3)            
        x_cat = torch.cat([x4, x1], dim=1)  
        out = self.dec2(x_cat)        
        return out


def create_diffusion_schedule(T, beta_start=1e-4, beta_end=0.02, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    annotations_file = os.path.join(args.data_dir, "annotations.csv")
    dataset = HeightmapDataset(annotations_file, args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    
    model = ConditionalUNet(time_emb_dim=128, cond_emb_dim=128, base_channels=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    T = args.T  
    betas, alphas, alphas_cumprod = create_diffusion_schedule(T, device=device)
    
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for images, cond in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)  
            cond = cond.to(device)      
            B = images.shape[0]
            
            t = torch.randint(0, T, (B,), device=device).long()
            alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(images).to(device)
            
            noisy_images = torch.sqrt(alpha_t) * images + torch.sqrt(1 - alpha_t) * noise
            optimizer.zero_grad()
            
            noise_pred = model(noisy_images, t, cond)
            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * B
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs} Loss: {epoch_loss:.4f}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "conditional_diffusion.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model for heightmap generation.")
    parser.add_argument("--data_dir", type=str, default="annotated_dataset2", help="Directory with images and annotations.csv")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion steps")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
