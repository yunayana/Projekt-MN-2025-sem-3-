import os
import random
from PIL import Image, ImageDraw
import torch
from torchvision import transforms, models
import requests
import json
import matplotlib.pyplot as plt

# Шляхи до даних
image_dir = "Foto"
labels_json_path = "etykieta.json"  # Файл з правильними етикетками

# Вибір випадкових 20 зображень
image_files = random.sample(os.listdir(image_dir), 20)

# Опис трансформацій для зображень
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Завантаження попередньо навчених моделей
models_to_test = {
    "ResNet18": models.resnet18(pretrained=True),
    "EfficientNetB0": models.efficientnet_b0(pretrained=True),
    "MobileNetV3": models.mobilenet_v3_large(pretrained=True),
    "VisionTransformer": models.vit_b_16(pretrained=True),
    "VGG16": models.vgg16(pretrained=True),
}

# Перехід моделей в режим оцінки
for model in models_to_test.values():
    model.eval()

# Завантаження етикеток ImageNet
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(url)
imagenet_labels = response.json()

# Завантаження правильних етикеток з файлу JSON
with open(labels_json_path, "r") as f:
    true_labels = json.load(f)

# Підготовка словника для зберігання прогнозів
predictions = {}

# Обробка зображень
for model_name, model in models_to_test.items():
    print(f"Обробка моделі: {model_name}")
    predictions[model_name] = {}  # Ініціалізація словника прогнозів для моделі
    plt.figure(figsize=(20, 10))  # Підготовка фігури для відображення зображень

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Додавання розміру пакету

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top3_prob, top3_catid = torch.topk(probabilities, 3)

            preds = {
                imagenet_labels[top3_catid[i]]: top3_prob[i].item()
                for i in range(3)
            }

        predictions[model_name][image_file] = preds

        # Відображення кожного зображення з прогнозами
        plt.subplot(4, 5, idx + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title("\n".join([f"{label}: {prob:.2f}" for label, prob in preds.items()]), fontsize=8)

    plt.suptitle(f"Прогнози для {model_name}", fontsize=16)
    plt.tight_layout()
    plt.show()

# Збереження прогнозів у файл JSON
with open("predictions.json", "w") as f:
    json.dump(predictions, f, indent=4)

# Завантаження збережених прогнозів
with open("predictions.json", "r") as f:
    predictions = json.load(f)

# Підготовка даних для порівняння (наприклад, точність або співпадіння топ-3)
accuracy_results = {model_name: 0 for model_name in models_to_test}
total_images = len(image_files)

# Створення словника для перекладу з польської на англійську
polish_to_english = {
    "widelec": "fork",
    "lyzka": "spoon",
    "noz": "knife",
    "paleczki": "chopsticks",
    "trzepaczka": "whisk",
    "Talerz": "plate",
    "Miska": "bowl",
    "Garnek": "pot",
    "Patelnia": "pan",
    "Kubek": "cup"
}

# Створення карти правильних етикеток для зображень
true_label_map = {}
for item in true_labels:
    true_label_map[item["image"]] = [polish_to_english.get(obj["class"].strip(), obj["class"].strip()) for obj in item["annotations"]]  # Переклад на англійську

# Порівняння прогнозів
for image_file, true_labels in true_label_map.items():
    for model_name, preds in predictions.items():
        if image_file in preds:  # Перевірка, чи є зображення в прогнозах
            predicted_labels = set(preds[image_file].keys())
            
            # Перевірка, чи є правильна етикетка серед прогнозів (Топ-3)
            if any(true_label in predicted_labels for true_label in true_labels):
                accuracy_results[model_name] += 1

# Нормалізація точності для отримання відсотків
accuracy_results = {model: accuracy / total_images for model, accuracy in accuracy_results.items()}

# Побудова графіка порівняння точності
plt.figure(figsize=(10, 6))
plt.bar(accuracy_results.keys(), accuracy_results.values(), color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Porownywanie dokładności modeli')
plt.ylim(0, 1)
plt.show()

# Виведення результатів
print("Wyniki porównania modeli:")
for model_name, accuracy in accuracy_results.items():
    print(f"{model_name}: {accuracy * 100:.2f}%")
