import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
from PIL import Image




def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(timesteps.device)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb




class SimpleUNet(nn.Module):
    def __init__(self, time_emb_dim=128, in_channels=1, out_channels=1, features=64):
        super(SimpleUNet, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(features, features*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(features*2, features*2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*2, features*2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(features*2, features, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(features*2, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb[:, :, None, None]
        
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.bottleneck(x2) + t_emb
        x4 = self.dec1(x3)
        x_cat = torch.cat([x4, x1], dim=1)
        out = self.dec2(x_cat)
        return out




T = 1000  
beta_start = 1e-5
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)




transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),   
    transforms.Lambda(lambda t: (t * 2) - 1)  
])


dataset = ImageFolder(root="dataset2", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SimpleUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
alphas_cumprod = alphas_cumprod.to(device)

epochs = 20  




def sample_image(model, shape, device, T):
    model.eval()
    with torch.no_grad():
        x = torch.randn(shape, device=device)
        for t in reversed(range(T)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            alpha_t = alphas_cumprod[t]
            beta_t = betas[t]
            noise = torch.randn_like(x).to(device) if t > 0 else 0
            noise_pred = model(x, t_tensor)
            x = (1 / torch.sqrt(alphas[t])) * (x - ((beta_t / torch.sqrt(1 - alpha_t)) * noise_pred))
            if t > 0:
                x = x + torch.sqrt(beta_t) * noise
        return x




print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to(device)
        batch_size = images.shape[0]
        t = torch.randint(0, T, (batch_size,), device=device).long()
        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(images).to(device)
        noisy_images = torch.sqrt(alpha_t) * images + torch.sqrt(1 - alpha_t) * noise
        optimizer.zero_grad()
        noise_pred = model(noisy_images, t)
        loss = criterion(noise_pred, noise)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_size
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}")

    
    if (epoch + 1) % 20 == 0:
        sample_batch = sample_image(model, (4, 1, 512, 512), device, T)
        sample_batch = (sample_batch + 1) / 2  
        sample_dir = os.path.join("generated4", f"epoch_{epoch+1:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        for i in range(sample_batch.shape[0]):
            img_array = (sample_batch[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode="L")
            img.save(os.path.join(sample_dir, f"heightmap_{i:03d}.png"))
        print(f"Sample images saved in {sample_dir}.")


torch.save(model.state_dict(), "diffusion_model4.pth")
print("Model saved.")
