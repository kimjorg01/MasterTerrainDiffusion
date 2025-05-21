import os
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from collections import Counter


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



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


def predict_image(model, image_path, transform, device, idx_to_label):
    """Loads an image, preprocesses it, and returns the predicted label and probabilities."""
    try:
        img = Image.open(image_path).convert('L') 
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0) 
        img_tensor = img_tensor.to(device)

        with torch.no_grad(): 
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0] 
            predicted_idx = torch.argmax(probabilities).item()
            predicted_label = idx_to_label.get(predicted_idx, f"Unknown Index {predicted_idx}") 

        
        prob_dict = {label_name: probabilities[idx].item() for idx, label_name in idx_to_label.items()}

        return predicted_label, prob_dict

    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        return None, None
    except Exception as e:
        logging.error(f"Failed to process or predict image {image_path}: {e}")
        return None, None


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    
    if not os.path.exists(args.model_path):
        logging.error(f"Model checkpoint not found: {args.model_path}")
        return

    logging.info(f"Loading checkpoint: {args.model_path}")
    
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except Exception as e:
         
         try:
             logging.warning(f"Loading with weights_only=False failed ({e}), attempting to load as state_dict...")
             state_dict = torch.load(args.model_path, map_location=device)
             checkpoint = {'model_state_dict': state_dict} 
         except Exception as e2:
             logging.error(f"Failed to load model checkpoint: {e2}")
             return


    
    
    if 'label_to_idx' in checkpoint:
        label_to_idx = checkpoint['label_to_idx']
        logging.info(f"Loaded label mapping from checkpoint: {label_to_idx}")
    else:
        
        label_to_idx = {"flat": 0, "mountain_ridges": 1, "mountain_rivers": 2}
        logging.warning(f"Label mapping not found in checkpoint, using default: {label_to_idx}")

    num_classes = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()} 

    
    try:
        model = SimpleCNNClassifier(num_classes=num_classes).to(device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
             
             model.load_state_dict(checkpoint)

        model.eval() 
        logging.info("Model loaded and set to evaluation mode.")
    except Exception as e:
        logging.error(f"Error instantiating or loading model state_dict: {e}")
        return

    
    
    
    infer_transform = transforms.Compose([
        transforms.ToTensor(),
        
    ])

    
    if args.image_path:
        
        logging.info(f"Predicting single image: {args.image_path}")
        predicted_label, probabilities = predict_image(model, args.image_path, infer_transform, device, idx_to_label)
        if predicted_label is not None:
            print(f"\nImage: {os.path.basename(args.image_path)}")
            print(f"  Predicted Class: {predicted_label}")
            print("  Probabilities:")
            for label, prob in probabilities.items():
                print(f"    {label}: {prob:.4f}")

    elif args.image_dir:
        
        if not os.path.isdir(args.image_dir):
            logging.error(f"Image directory not found or is not a directory: {args.image_dir}")
            return

        logging.info(f"Predicting images in directory: {args.image_dir}")
        image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        if not image_files:
            logging.warning(f"No image files found in directory: {args.image_dir}")
            return

        predictions = []
        class_counts = Counter()

        for filename in tqdm(image_files, desc="Predicting Images"):
            image_path = os.path.join(args.image_dir, filename)
            predicted_label, _ = predict_image(model, image_path, infer_transform, device, idx_to_label) 
            if predicted_label is not None:
                predictions.append({'filename': filename, 'predicted_label': predicted_label})
                class_counts[predicted_label] += 1
            else:
                 predictions.append({'filename': filename, 'predicted_label': 'Error'})


        
        print("\n--- Prediction Results ---")
        
        
        

        
        print("\nSummary Counts:")
        total_processed = len(predictions)
        total_errors = sum(1 for p in predictions if p['predicted_label'] == 'Error')
        print(f"  Processed: {total_processed} files")
        print(f"  Errors: {total_errors}")
        print("  Predicted Class Counts:")
        for label, count in sorted(class_counts.items()):
            print(f"    {label}: {count}")
        print("-------------------------")

    else:
        logging.error("No input specified. Please provide --image_path or --image_dir.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify heightmap images using a trained CNN model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained classifier model checkpoint (.pth file).")

    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_path", type=str, help="Path to a single image file to classify.")
    input_group.add_argument("--image_dir", type=str, help="Path to a directory containing images to classify.")

    args = parser.parse_args()
    main(args)