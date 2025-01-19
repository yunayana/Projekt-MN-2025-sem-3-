import os 
import numpy as np 
from PIL import Image 
  
import matplotlib.pyplot as plt 
  
import torch 
import torchvision

class CustomDataset(Dataset):
    def __init__(self, annotations, image_dir, transform=None):
        self.annotations = annotations
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = os.path.join(self.image_dir, annotation['image'])
        image = Image.open(image_path).convert("RGB")

        labels = [0] * len(class_names)
        for obj in annotation['objects']:
            if obj['class'] in class_names:
                labels[class_names.index(obj['class'])] = 1

        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels, image_path 