import json
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Funkcja do wczytania danych z pliku JSON
def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# Funkcja do wizualizacji granic obiektów na obrazie
def visualize_borders(image_path, objects):
    # Wczytaj obraz za pomocą OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konwertowanie do RGB dla matplotlib
    
    # Utwórz wykres za pomocą matplotlib
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    # Przejdź przez obiekty i narysuj granice
    for obj in objects:
        x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
        # Narysuj prostokąt na obrazie
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none'))
        # Opcjonalnie dodaj nazwę klasy obiektu
        ax.text(x1, y1 - 10, obj['class'], color='red', fontsize=12, backgroundcolor='white')
    
    # Wyświetl obraz z nałożonymi granicami
    plt.show()

# Funkcja do sprawdzenia i wizualizacji granic dla wszystkich zdjęć w folderze
def check_borders(json_file, images_folder):
    # Wczytaj dane z pliku JSON
    data = load_json(json_file)
    
    # Sprawdź każdy obraz i narysuj granice
    for entry in data:
        image_name = entry.get('image')  # Nazwa obrazu w danych JSON
        if image_name:
            image_path = os.path.join(images_folder, image_name)
            if os.path.exists(image_path):
                objects = entry.get('objects', [])  # Lista obiektów
                visualize_borders(image_path, objects)
            else:
                print(f"Obraz {image_name} nie istnieje w folderze.")
        else:
            print("Brak nazwy obrazu w danych JSON.")

# Przykładowe użycie:
json_file = 'labels.json'  # Podaj ścieżkę do pliku JSON
images_folder = 'Foto'  # Podaj ścieżkę do folderu z obrazami

check_borders(json_file, images_folder)
