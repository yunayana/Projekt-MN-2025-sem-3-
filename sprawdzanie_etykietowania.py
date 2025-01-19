import json
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def visualize_borders(image_path, annotations):
    # Wczytaj obraz za pomocą OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    # przechodzenie przez obiekty i rysowanie granic
    for annotation in annotations:
        class_name = annotation['class']
        bbox = annotation['bbox']
        x1, y1, x2, y2 = bbox
 
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none'))
        ax.text(x1, y1 - 10, class_name.strip(), color='red', fontsize=12, backgroundcolor='white')
    plt.show()

# funkcja sprawdzenia i wizualizacji granic na zdjeciach w folderze
def check_borders(json_file, images_folder):
    data = load_json(json_file)
    
    #sprawdzenie kazdego obrazu i rysowanie granicy
    for entry in data:
        image_name = entry.get('image')  
        if image_name:
            image_path = os.path.join(images_folder, image_name)
            if os.path.exists(image_path):
                annotations = entry.get('annotations', []) 
                visualize_borders(image_path, annotations)
            else:
                print(f"Obraz {image_name} nie istnieje w folderze.")
        else:
            print("Brak nazwy obrazu w danych JSON.")

# Przykładowe użycie:
json_file = 'etykieta.json' 
images_folder = 'Foto' 

check_borders(json_file, images_folder)
