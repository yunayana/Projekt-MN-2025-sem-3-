import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        """
        :param json_file: Ścieżka do pliku JSON z etykietami
        :param img_dir: Katalog, w którym znajdują się obrazy
        :param transform: Transformacje, które mają zostać zastosowane do obrazów
        """
        self.img_dir = img_dir
        self.transform = transform
        # Wczytanie pliku JSON
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        # Zwraca liczbę obrazów w zbiorze
        return len(self.data)

    def __getitem__(self, idx):
        # Wczytanie obrazu
        img_name = os.path.join(self.img_dir, self.data[idx]['image'])
        image = Image.open(img_name).convert("RGB")  # Konwertuj na RGB jeśli obraz jest w innym formacie

        boxes = []
        labels = []
        
        for obj in self.data[idx]['objects']:
            # Przygotowanie współrzędnych bounding boxów i klas
            boxes.append([obj['x1'], obj['y1'], obj['x2'], obj['y2']])
            labels.append(obj['class'])
        
        # Konwersja współrzędnych do formatu Tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        # Konwersja etykiet klas do numerów (np. używając słownika)
        class_to_idx = {
            "widelec": 0,
            "lyzka": 1,
            "noz": 2,
            "talerz": 3,
            "miseczka": 4,
            "paleczki": 5,
            "trzepaczka": 6,
            "garnek": 7,
            "patelnia": 8,
            "kubek": 9  # Dodaj klasę 'talerz' z odpowiednim numerem
        }
        
        labels = [class_to_idx[label] for label in labels]
        labels = torch.tensor(labels, dtype=torch.long)

        # Przygotowanie słownika
        sample = {'image': image, 'boxes': boxes, 'labels': labels}

        # Zastosowanie transformacji tylko na obraz (nie na bounding boxy)
        if self.transform:
            image = self.transform(image)
            sample['image'] = image

        return sample


# Funkcja do łączenia batcha
def collate_fn(batch):
    images = []
    boxes = []
    labels = []

    # Find max number of boxes/labels in a batch
    max_num_boxes = max([len(b['boxes']) for b in batch])
    
    for b in batch:
        images.append(b['image'])
        
        # Padding na liczbie boxów (jeśli trzeba)
        box_padding = max_num_boxes - len(b['boxes'])
        box_padded = torch.cat([b['boxes'], torch.zeros(box_padding, b['boxes'].size(1))], dim=0)
        boxes.append(box_padded)
        
        # Padding na liczbie etykiet (jeśli trzeba)
        label_padding = max_num_boxes - len(b['labels'])
        label_padded = torch.cat([b['labels'], torch.tensor([-1]*label_padding)], dim=0)
        labels.append(label_padded)

    # Stosowanie na obrazie i pozostałych danych
    images = torch.stack(images)
    boxes = torch.stack(boxes)
    labels = torch.stack(labels)

    return {'image': images, 'boxes': boxes, 'labels': labels}


# Transformacje (opcjonalnie)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Zmienia rozmiar obrazu na 256x256
    transforms.ToTensor(),  # Konwertuje obraz na Tensor
])

# Ścieżki do pliku JSON i katalogu z obrazami
json_file = r"C:\Users\user\Desktop\projekt_mn_ja_ostatni\labels.json"
img_dir = r"C:\Users\user\Desktop\projekt_mn_ja_ostatni\Foto"

# Załaduj dataset
dataset = CustomDataset(json_file=json_file, img_dir=img_dir, transform=transform)

# Załaduj dane za pomocą DataLoader z naszą własną funkcją collate_fn
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Przykład eksportu danych do pliku JSON
export_data = []

# Iteracja po danych
for i, sample in enumerate(dataloader):
    images = sample['image']
    boxes = sample['boxes']
    labels = sample['labels']
    
    for j in range(len(images)):
        image_data = {
            'image': f'image_{i*len(images) + j}.jpg',  # Nazwa obrazu (możesz dostosować)
            'boxes': boxes[j].tolist(),  # Współrzędne boxów
            'labels': labels[j].tolist()  # Etykiety klas
        }
        export_data.append(image_data)

    print(f"Batch {i+1}:")
    print(f"Images shape: {images.shape}")
    print(f"Boxes: {boxes}")
    print(f"Labels: {labels}")
    # Możesz tu wykonać dalsze operacje na obrazie i etykietach

# Zapisz dane do pliku JSON
output_file = r"C:\Users\user\Desktop\projekt_mn_ja_ostatni\exported_data.json"
with open(output_file, 'w') as f:
    json.dump(export_data, f, indent=4)

print(f"Dane zostały zapisane do {output_file}")
