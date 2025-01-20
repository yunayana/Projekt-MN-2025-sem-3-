import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
import os

class ImageLabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeling Tool")
        self.root.configure(bg="#2E2E2E")

        # Utwórz canvas, na którym będziemy rysować obraz i prostokąty
        self.label_canvas = tk.Canvas(root, bg="#1E1E1E", highlightthickness=0)
        self.label_canvas.pack(fill=tk.BOTH, expand=True)

        # Przyciski do ładowania obrazów, zapisywania i resetowania
        self.load_button = tk.Button(root, text="Load Images", command=self.load_images, bg="#4A4A4A", fg="white", font=("Arial", 12))
        self.load_button.pack(side=tk.LEFT, padx=10)

        self.save_next_button = tk.Button(root, text="Save & Next", command=self.save_and_next, bg="#4A4A4A", fg="white", font=("Arial", 12))
        self.save_next_button.pack(side=tk.RIGHT, padx=10)

        self.reset_button = tk.Button(root, text="Reset Current Labels", command=self.reset_labels, bg="#4A4A4A", fg="white", font=("Arial", 12))
        self.reset_button.pack(side=tk.RIGHT, padx=10)

        # Dodanie dużego napisu z wyborem klasy
        self.class_label = tk.Label(root, text="Select Class:", bg="#2E2E2E", fg="white", font=("Arial", 14))
        self.class_label.pack(side=tk.LEFT, padx=10)

        self.class_options = ["widelec", "lyzka", "noz", "paleczki", "trzepaczka", "Talerz", "Miska", "Garnek", "Patelnia", "Kubek"]
        self.class_var = tk.StringVar(value=self.class_options[0])

        # Dodanie listy rozwijanej do wyboru klasy
        self.class_dropdown = ttk.Combobox(root, textvariable=self.class_var, values=self.class_options, state="readonly", font=("Arial", 12))
        self.class_dropdown.pack(side=tk.LEFT, padx=10)

        # Inicjalizacja zmiennych do obsługi obrazów
        self.image_index = 0
        self.images = []
        self.labels = []
        self.current_image = None
        self.bounding_boxes = []
        self.start_x = self.start_y = 0

        # Obsługa rysowania prostokątów na canvasie
        self.label_canvas.bind("<Button-1>", self.start_draw)
        self.label_canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.label_canvas.bind("<ButtonRelease-1>", self.finish_draw)

        # Ścieżka, gdzie będziemy zapisywać dane
        self.output_path = os.path.join(os.path.dirname(__file__), "labels.json")
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r") as json_file:
                    content = json_file.read().strip()
                    if content:
                        self.labels = json.loads(content)
                    else:
                        self.labels = []
            except json.JSONDecodeError:
                self.labels = []

    def load_images(self):
        """Ładuje obrazy z wybranego folderu."""
        folder_path = filedialog.askdirectory()  # Poproś o wybór folderu
        if folder_path:
            # Załaduj obrazy z folderu
            self.images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if self.images:
                self.image_index = 0
                self.load_image()  # Załaduj pierwszy obraz
            else:
                messagebox.showerror("No Images", "No images found in the selected folder.")  # Komunikat, jeśli nie znaleziono obrazów

    def load_image(self):
        """Ładuje i wyświetla obraz w canvasie."""
        if 0 <= self.image_index < len(self.images):
            try:
                image_path = self.images[self.image_index]
                self.current_image = Image.open(image_path)
                self.current_image.thumbnail((self.label_canvas.winfo_width(), self.label_canvas.winfo_height()), Image.Resampling.LANCZOS)  # Używamy LANCZOS
                self.tk_image = ImageTk.PhotoImage(self.current_image)
                self.label_canvas.delete("all")  # Usuń poprzedni obraz

                canvas_width = self.label_canvas.winfo_width()
                canvas_height = self.label_canvas.winfo_height()
                image_width, image_height = self.tk_image.width(), self.tk_image.height()
                x_offset = (canvas_width - image_width) // 2
                y_offset = (canvas_height - image_height) // 2

                # Wyświetlenie obrazu na canvasie
                self.label_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.tk_image)
                self.bounding_boxes = []
                self.root.title(f"Labeling Image {self.image_index + 1}/{len(self.images)}: {os.path.basename(image_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")  # Obsługuje błędy podczas ładowania obrazu

    def reset_labels(self):
        """Resetuje obecne etykiety i ładuje ponownie obraz."""
        self.bounding_boxes.clear()
        self.load_image()

    def start_draw(self, event):
        """Zaczyna rysowanie prostokąta od miejsca kliknięcia."""
        self.start_x, self.start_y = event.x, event.y

    def draw_rectangle(self, event):
        """Rysuje prostokąt na canvasie w trakcie przeciągania myszy."""
        self.label_canvas.delete("temp_rect")  # Usuń poprzedni prostokąt
        self.label_canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline="red", fill="yellow", stipple="gray12", tag="temp_rect"
        )

    def finish_draw(self, event):
        """Kończy rysowanie prostokąta po zwolnieniu przycisku myszy."""
        end_x, end_y = event.x, event.y
        self.label_canvas.delete("temp_rect")  # Usuń tymczasowy prostokąt
        rect_id = self.label_canvas.create_rectangle(
            self.start_x, self.start_y, end_x, end_y,
            outline="blue", fill="lightblue", stipple="gray25"
        )
        label_text = f"{self.class_var.get()}"
        text_x = (self.start_x + end_x) / 2
        text_y = (self.start_y + end_y) / 2
        self.label_canvas.create_text(text_x, text_y, text=label_text, fill="black")
        self.bounding_boxes.append({"x1": self.start_x, "y1": self.start_y, "x2": end_x, "y2": end_y, "class": self.class_var.get()})
        self.save_labels()

    def save_labels(self):
        """Zapisuje etykiety do pliku JSON."""
        label_data = {
            "image": os.path.basename(self.images[self.image_index]),
            "objects": self.bounding_boxes
        }
        # Sprawdź, czy dane dla obrazu już istnieją, jeśli tak, zaktualizuj je
        existing_entry = next((entry for entry in self.labels if entry["image"] == label_data["image"]), None)
        if existing_entry:
            self.labels.remove(existing_entry)
        self.labels.append(label_data)
        try:
            with open(self.output_path, "w") as json_file:
                json.dump(self.labels, json_file, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save labels: {e}")  # Obsługuje błędy zapisu

    def save_and_next(self):
        """Zapisuje etykiety i przechodzi do następnego obrazu."""
        self.save_labels()
        self.image_index += 1
        if self.image_index < len(self.images):
            self.load_image()  # Załaduj następny obraz
        else:
            messagebox.showinfo("End", "No more images to label.")  # Komunikat, jeśli brak obrazów

if __name__ == "__main__":
    root = tk.Tk()
    tool = ImageLabelingTool(root)
    root.mainloop()
