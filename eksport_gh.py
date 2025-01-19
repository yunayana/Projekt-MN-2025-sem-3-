
#fragment wyciagniety z ModelTraining
import os 
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# llista klas (10)
class_names = ['widelec', 'lyzka', 'noz', 'paleczki', 'trzepaczka', 'Talerz', 'Miska', 'Garnek', 'Patelnia', 'Kubek']

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# fragment z przykladu https://www.geeksforgeeks.org/image-datasets-dataloaders-and-transforms-in-pytorch/
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



# sciezka do zdjec
image_dir = "/path/to/images"

dataset = CustomDataset(annotations, image_dir, transform=transform)

image, labels, path = dataset[0]
print(f"Image path: {path}")
print(f"Labels: {labels}")
