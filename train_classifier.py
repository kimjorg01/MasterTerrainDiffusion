import os
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from tqdm import tqdm
import logging
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HeightmapClassifDataset(Dataset):
    def __init__(self, annotations_df, images_dir, label_to_idx, transform=None):
        """
        Args:
            annotations_df (pd.DataFrame): DataFrame containing 'filename' and 'label' for this split.
            images_dir (str): Directory containing the images.
            label_to_idx (dict): Mapping from label name to integer index.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations_df = annotations_df.reset_index(drop=True) 
        self.images_dir = images_dir
        self.transform = transform
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.annotations_df.iloc[idx]
        img_name = os.path.join(self.images_dir, row['filename'])
        label_name = row['label']

        try:
            image = Image.open(img_name).convert('L') 
            label_idx = self.label_to_idx[label_name]
            label = torch.tensor(label_idx, dtype=torch.long)

            if self.transform:
                image = self.transform(image)

            
            if not isinstance(image, torch.Tensor):
                 
                 image = transforms.ToTensor()(image)

            return image, label

        except FileNotFoundError:
            logging.warning(f"Image not found: {img_name}. Skipping.")
            
            return None
        except Exception as e:
            logging.error(f"Error loading image {img_name} or label {label_name}: {e}")
            return None


def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        
        
        return torch.empty((0, 1, 256, 256)), torch.empty((0,), dtype=torch.long)
    return torch.utils.data.dataloader.default_collate(batch)


class SimpleCNNClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNNClassifier, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 

             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        self.flattened_size = 512 * 8 * 8

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
            nn.Linear(1024, num_classes)
            
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    
    annotations_path = os.path.join(args.data_dir, "annotations.csv")
    if not os.path.exists(annotations_path):
        logging.error(f"Annotations file not found: {annotations_path}")
        return
    logging.info(f"Loading annotations from: {annotations_path}")
    try:
        full_df = pd.read_csv(annotations_path)
        
        if not {'filename', 'label'}.issubset(full_df.columns):
             logging.error("Annotations CSV must contain 'filename' and 'label' columns.")
             return
        logging.info(f"Loaded {len(full_df)} total records.")
    except Exception as e:
        logging.error(f"Error reading CSV {annotations_path}: {e}")
        return

    
    known_labels = ["flat", "mountain_ridges", "mountain_rivers"] 
    label_to_idx = {label: i for i, label in enumerate(known_labels)}
    num_classes = len(known_labels)
    logging.info(f"Class mapping: {label_to_idx}")

    
    full_df = full_df[full_df['label'].isin(known_labels)]
    logging.info(f"Filtered to {len(full_df)} records with known labels.")

    
    train_df_list = []
    val_df_list = []
    required_total = args.train_samples_per_class + args.val_samples_per_class

    logging.info(f"Performing {args.train_samples_per_class}/{args.val_samples_per_class} train/validation split per class...")
    for label in known_labels:
        label_df = full_df[full_df['label'] == label]
        n_label = len(label_df)
        logging.info(f"Found {n_label} images for class '{label}'.")

        if n_label < required_total:
            logging.error(f"Not enough images for class '{label}'. Found {n_label}, need {required_total}. Cannot perform split.")
            return

        
        train_subset, val_subset = train_test_split(
            label_df,
            train_size=args.train_samples_per_class,
            test_size=args.val_samples_per_class,
            random_state=args.seed, 
            shuffle=True 
        )
        train_df_list.append(train_subset)
        val_df_list.append(val_subset)
        logging.info(f"  Split for '{label}': {len(train_subset)} train, {len(val_subset)} val.")

    train_df = pd.concat(train_df_list).sample(frac=1, random_state=args.seed).reset_index(drop=True) 
    val_df = pd.concat(val_df_list).sample(frac=1, random_state=args.seed).reset_index(drop=True) 

    logging.info(f"Total training samples: {len(train_df)}")
    logging.info(f"Total validation samples: {len(val_df)}")
    logging.info(f"Training class distribution:\n{train_df['label'].value_counts()}")
    logging.info(f"Validation class distribution:\n{val_df['label'].value_counts()}")

    
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        
        
        
        transforms.ToTensor(), 
        
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        
    ])

    
    train_dataset = HeightmapClassifDataset(train_df, args.data_dir, label_to_idx, transform=train_transform)
    val_dataset = HeightmapClassifDataset(val_df, args.data_dir, label_to_idx, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)

    
    model = SimpleCNNClassifier(num_classes=num_classes).to(device)
    logging.info(f"Classifier model created with ~{sum(p.numel() for p in model.parameters()):,} parameters.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    
    best_val_acc = 0.0
    best_epoch = -1
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_classifier_model.pth")

    logging.info("Starting classifier training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)

        for inputs, labels in progress_bar:
            
            if inputs.numel() == 0: continue

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                 if inputs.numel() == 0: continue
                 inputs, labels = inputs.to(device), labels.to(device)
                 outputs = model(inputs)
                 loss = criterion(outputs, labels)
                 val_loss += loss.item() * inputs.size(0)
                 _, predicted = torch.max(outputs.data, 1)
                 val_total += labels.size(0)
                 val_correct += (predicted == labels).sum().item()

        
        if val_total > 0:
            epoch_val_loss = val_loss / val_total
            epoch_val_acc = 100.0 * val_correct / val_total
        else:
            epoch_val_loss = 0.0
            epoch_val_acc = 0.0
            logging.warning("Validation set evaluation skipped: val_total is zero.")


        logging.info(f"Epoch {epoch+1}/{args.epochs} - "
                     f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% - "
                     f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

        
        scheduler.step(epoch_val_acc) 

        
        if epoch_val_acc > best_val_acc:
            logging.info(f"Validation accuracy improved ({best_val_acc:.2f}% -> {epoch_val_acc:.2f}%). Saving model...")
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1
            try:
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_val_loss,
                    'accuracy': best_val_acc,
                    'label_to_idx': label_to_idx 
                }, best_model_path)
                logging.info(f"Best model saved to {best_model_path}")
            except Exception as e:
                logging.error(f"Error saving best model: {e}")

    logging.info(f"Training finished. Best validation accuracy: {best_val_acc:.2f}% achieved at epoch {best_epoch}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN classifier for heightmap types.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images and annotations.csv.")
    parser.add_argument("--save_dir", type=str, default="classifier_model", help="Directory to save the best trained model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--train_samples_per_class", type=int, default=4000, help="Number of training samples per class.")
    parser.add_argument("--val_samples_per_class", type=int, default=1000, help="Number of validation samples per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility of splits.")

    args = parser.parse_args()
    train_classifier(args)