import json
import os
import random
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import cv2

# Список классов
class_names = ['widelec', 'lyzka', 'noz', 'paleczki', 'trzepaczka', 'Talerz', 'Miska', 'Garnek', 'Patelnia', 'Kubek']

# Параметры
annotations_path = 'etykieta.json'  # Путь к аннотациям
image_dir = 'Foto'  # Папка с изображениями

# Функция для загрузки данных из JSON
def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Класс для набора данных
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

        # Массив меток для каждого класса (многоклассовая бинарная классификация)
        labels = [0] * len(class_names)
        for obj in annotation['annotations']:
            if obj['class'] in class_names:
                labels[class_names.index(obj['class'])] = 1

        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels, image_path

# Функция для тренировки модели
def train_model(model, dataset, criterion, optimizer, num_epochs=15):
    model.train()
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = len(train_loader)

        for i, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Вывод прогресса
            progress = (i + 1) / total_batches * 100
            print(f'\rEpoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{total_batches}] Loss: {loss.item():.4f} Progress: {progress:.2f}%', end="")

        print(f"\nEpoch {epoch+1}/{num_epochs}, Loss: {running_loss/total_batches:.4f}")

# Функция для предсказания
def predict_with_model(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(image_tensor)
        predicted_probs = torch.sigmoid(outputs)  # Вероятности для каждого класса
        predicted_labels = predicted_probs > 0.5  # Повышаем порог до 0.5 для лучшего отделения классов
    
    return predicted_labels, predicted_probs

# Функция для визуализации рамок
def visualize_borders(image_path, annotations, predicted_labels=None, predicted_probs=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Конвертируем в RGB для matplotlib

    fig, ax = plt.subplots()
    ax.imshow(image)

    # Визуализируем аннотации из JSON
    for annotation in annotations:
        class_name = annotation['class']
        bbox = annotation['bbox']
        x1, y1, x2, y2 = bbox

        # Рисуем рамку для аннотации
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none'))
        ax.text(x1, y1 - 10, class_name.strip(), color='red', fontsize=12, backgroundcolor='white')

    # Визуализируем предсказания модели
    if predicted_labels is not None and predicted_probs is not None:
        for i in range(len(predicted_labels[0])):
            if predicted_labels[0][i].item():
                prob = predicted_probs[0][i].item()
                class_name = class_names[i]
                bbox = [50, 50, 200, 200]  # Примерные координаты для демонстрации
                x1, y1, x2, y2 = bbox
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none'))
                ax.text(x1, y1 - 10, f"{class_name} ({prob:.2f})", color='green', fontsize=12, backgroundcolor='white')

    plt.show()

# Функция для проверки и визуализации 20 случайных изображений
def check_borders(json_file, images_folder, model, transform):
    data = load_json(json_file)
    
    # Получаем 20 случайных изображений для предсказаний
    random_entries = random.sample(data, 20)

    for entry in random_entries:
        image_name = entry.get('image')
        if image_name:
            image_path = os.path.join(images_folder, image_name)
            if os.path.exists(image_path):
                annotations = entry.get('annotations', [])
                predicted_labels, predicted_probs = predict_with_model(image_path, model)
                visualize_borders(image_path, annotations, predicted_labels, predicted_probs)
            else:
                print(f"Изображение {image_name} не найдено.")
        else:
            print("Отсутствует название изображения в данных JSON.")

# Функция для расчета точности по каждому классу
def calculate_class_accuracy(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    correct = np.zeros(len(class_names))
    total = np.zeros(len(class_names))

    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = predicted_probs > 0.5

            for i in range(len(class_names)):
                total[i] += labels[0, i].item()
                if labels[0, i].item() == 1 and predicted_labels[0, i].item() == 1:
                    correct[i] += 1

    # Вычисляем точность для каждого класса
    class_accuracies = correct / total * 100  # В процентах
    return class_accuracies

# Функция для построения графика точности по классам
def plot_class_accuracies(accuracies):
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, accuracies, color='skyblue', edgecolor='black')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of individual classes')

    # Добавляем текстовые метки на график
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 1, f'{acc:.2f}%', ha='center', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

# Инициализация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  # Подгонка под количество классов
model = model.to(device)

# Критерий и оптимизатор
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Загружаем данные и создаем набор для обучения
data = load_json(annotations_path)
dataset = CustomDataset(data, image_dir, transform)

# Тренируем модель (15 эпох)
train_model(model, dataset, criterion, optimizer, num_epochs=15)

# Расчет точности по классам и построение графика
class_accuracies = calculate_class_accuracy(model, dataset)
plot_class_accuracies(class_accuracies)

# Проверяем и визуализируем предсказания на 20 случайных изображениях
check_borders(annotations_path, image_dir, model, transform)
