#program innego uzytkownika

import os
import sys
import json
import logging
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGraphicsScene,
    QGraphicsView,
    QComboBox,
    QWidget,
    QGraphicsRectItem,
    QSlider,
    QStatusBar,
    QGridLayout,
    QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPen
from PyQt5.QtCore import Qt


logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')


class Etykieta(QGraphicsRectItem):
    def __init__(self, x, y, szerokosc, wysokosc, kolor):
        super().__init__(x, y, szerokosc, wysokosc)
        self.setPen(QPen(QColor(kolor), 2))
        self.setBrush(QColor(kolor).lighter(160))
        self.setOpacity(0.5)


def load_annotations(filepath):
    """
    Wczytuje istniejące adnotacje z pliku JSON.
    Zwraca listę adnotacji lub pustą listę, jeśli plik nie istnieje lub jest pusty.
    """
    if os.path.exists(filepath):
        if os.path.getsize(filepath) > 0:
            with open(filepath, 'r') as f:
                return json.load(f)
    return []


def save_annotations(filepath, annotations_data):
    """
    Zapisuje adnotacje do pliku JSON w formacie z wcięciem 4 spacji.
    """
    with open(filepath, 'w') as f:
        json.dump(annotations_data, f, indent=4)


class AplikacjaOznaczaniaObrazow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikacja do Oznaczania Obrazów")
        self.setGeometry(100, 100, 800, 600)

        try:
            self.folder_z_obrazami = QFileDialog.getExistingDirectory(self, "Wybierz folder z obrazami")
            if not self.folder_z_obrazami:
                raise FileNotFoundError("Nie wybrano folderu z obrazami.")
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Błąd", str(e))
            sys.exit(1)

        self.lista_obrazow = [
            f for f in os.listdir(self.folder_z_obrazami)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not self.lista_obrazow:
            QMessageBox.warning(self, "Błąd", "Folder nie zawiera plików PNG/JPG/JPEG.")
            sys.exit(1)

        self.liczba_obrazow = len(self.lista_obrazow)
        self.indeks_obecnego_obrazu = 0
        self.etykiety = []

        self.plik_adnotacji = os.path.join(os.getcwd(), "adnotacje.json")
        self.klasy_pojazdow = ['samochod', 'brak']

        self.kolory = ['#00FF99', '#FF6600', '#3366FF', '#FF0066', '#33FFCC', '#9900FF', '#CCCC00', '#FFCCFF']
        self.indeks_obecnego_koloru = 0

        self.istniejace_dane = load_annotations(self.plik_adnotacji)

        self.inicjalizuj_UI()

    def inicjalizuj_UI(self):
        glowny_uklad = QVBoxLayout()

        gorny_uklad = QHBoxLayout()
        gorne_tlo = QWidget(self)
        gorne_tlo.setStyleSheet("background-color: #2C3E50; padding: 10px;")
        gorne_tlo.setLayout(gorny_uklad)

        naglowek = QLabel("Menu Opcji", self)
        naglowek.setStyleSheet("font-size: 16px; font-weight: bold; color: #ECF0F1; margin-right: 10px;")
        gorny_uklad.addWidget(naglowek)

        self.wybor_klasy = QComboBox(self)
        self.wybor_klasy.addItems(self.klasy_pojazdow)
        self.wybor_klasy.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #BDC3C7;
                border-radius: 5px;
                background-color: #34495E;
                color: #ECF0F1;
            }
            QComboBox::drop-down {
                border: 0px;
            }
        """)
        gorny_uklad.addWidget(self.wybor_klasy)

        self.przycisk_dodaj_etykieta = QPushButton("Dodaj Etykieta", self)
        self.przycisk_dodaj_etykieta.setStyleSheet("""
            QPushButton {
                background-color: #1ABC9C;
                color: #FFFFFF;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #16A085;
            }
        """)
        self.przycisk_dodaj_etykieta.clicked.connect(self.dodaj_etykieta)
        gorny_uklad.addWidget(self.przycisk_dodaj_etykieta)

        self.przycisk_resetuj = QPushButton("Resetuj", self)
        self.przycisk_resetuj.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: #FFFFFF;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        self.przycisk_resetuj.clicked.connect(self.resetuj_etykiety)
        gorny_uklad.addWidget(self.przycisk_resetuj)

        self.przycisk_zapisz = QPushButton("Zapisz", self)
        self.przycisk_zapisz.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: #FFFFFF;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)
        self.przycisk_zapisz.clicked.connect(self.zapisz_adnotacje)
        gorny_uklad.addWidget(self.przycisk_zapisz)

        self.przycisk_zakoncz = QPushButton("Zakończ", self)
        self.przycisk_zakoncz.setStyleSheet("""
            QPushButton {
                background-color: #9B59B6;
                color: #FFFFFF;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #8E44AD;
            }
        """)
        self.przycisk_zakoncz.clicked.connect(self.close)
        gorny_uklad.addWidget(self.przycisk_zakoncz)

        glowny_uklad.addWidget(gorne_tlo)

        self.etykieta_licznika_obrazow = QLabel(
            f"Obraz: {self.indeks_obecnego_obrazu + 1} / {self.liczba_obrazow}",
            self
        )
        self.etykieta_licznika_obrazow.setStyleSheet("margin-top: 5px; color: #34495E; font-weight: bold;")
        glowny_uklad.addWidget(self.etykieta_licznika_obrazow)

        self.scena = QGraphicsScene(self)
        self.widok = QGraphicsView(self.scena)
        self.widok.setAlignment(Qt.AlignCenter)

        self.current_scale = 1.0
        self.min_scale = 0.5
        self.max_scale = 2.0

        zoom_layout = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.setStyleSheet("background-color: #27AE60; color: #FFFFFF; padding: 5px; border-radius: 5px;")
        self.zoom_in_btn.clicked.connect(self.zoom_in)

        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.setStyleSheet("background-color: #F39C12; color: #FFFFFF; padding: 5px; border-radius: 5px;")
        self.zoom_out_btn.clicked.connect(self.zoom_out)

        zoom_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addWidget(self.zoom_out_btn)
        glowny_uklad.addLayout(zoom_layout)

        self.slider_y1 = QSlider(Qt.Vertical)
        self.slider_y1.setMinimum(0)
        self.slider_y1.setMaximum(1000)
        self.slider_y1.setInvertedAppearance(True)
        self.slider_y1.setTickInterval(1)
        self.slider_y1.setSingleStep(1)
        self.slider_y1.setStyleSheet("""
            QSlider::handle:vertical {
                background-color: #3498DB;
                border: 1px solid #2980B9;
                height: 10px;
                margin: 0 -2px;
            }
        """)
        self.slider_y1.valueChanged.connect(self.aktualizuj_prostokat)

        self.slider_y2 = QSlider(Qt.Vertical)
        self.slider_y2.setMinimum(0)
        self.slider_y2.setMaximum(1000)
        self.slider_y2.setInvertedAppearance(True)
        self.slider_y2.setTickInterval(1)
        self.slider_y2.setSingleStep(1)
        self.slider_y2.setStyleSheet(self.slider_y1.styleSheet())
        self.slider_y2.valueChanged.connect(self.aktualizuj_prostokat)

        self.slider_x1 = QSlider(Qt.Horizontal)
        self.slider_x1.setMinimum(0)
        self.slider_x1.setMaximum(1000)
        self.slider_x1.setTickInterval(1)
        self.slider_x1.setSingleStep(1)
        self.slider_x1.setStyleSheet("""
            QSlider::handle:horizontal {
                background-color: #1ABC9C;
                border: 1px solid #16A085;
                width: 10px;
                margin: -2px 0;
            }
        """)
        self.slider_x1.valueChanged.connect(self.aktualizuj_prostokat)

        self.slider_x2 = QSlider(Qt.Horizontal)
        self.slider_x2.setMinimum(0)
        self.slider_x2.setMaximum(1000)
        self.slider_x2.setTickInterval(1)
        self.slider_x2.setSingleStep(1)
        self.slider_x2.setStyleSheet(self.slider_x1.styleSheet())
        self.slider_x2.valueChanged.connect(self.aktualizuj_prostokat)

        suwak_uklad = QGridLayout()
        suwak_uklad.addWidget(self.widok, 0, 0, 3, 3)
        suwak_uklad.addWidget(self.slider_y1, 0, 3, 1, 1)
        suwak_uklad.addWidget(self.slider_y2, 1, 3, 1, 1)
        suwak_uklad.addWidget(self.slider_x1, 3, 0, 1, 3)
        suwak_uklad.addWidget(self.slider_x2, 4, 0, 1, 3)

        glowny_uklad.addLayout(suwak_uklad)

        glowny_widget = QWidget()
        glowny_widget.setLayout(glowny_uklad)
        self.setCentralWidget(glowny_widget)

        self.setStatusBar(QStatusBar(self))

        self.zaladuj_obraz()

    def zoom_in(self):
        """Powiększ widok, ale tylko do maksymalnego limitu."""
        if self.current_scale < self.max_scale:
            self.widok.scale(1.2, 1.2)
            self.current_scale *= 1.2
            if self.current_scale > self.max_scale:
                self.current_scale = self.max_scale

    def zoom_out(self):
        """Pomniejsz widok, ale tylko do minimalnego limitu."""
        if self.current_scale > self.min_scale:
            self.widok.scale(0.8, 0.8)
            self.current_scale *= 0.8
            if self.current_scale < self.min_scale:
                self.current_scale = self.min_scale

    def zaladuj_obraz(self):
        """Wczytuje kolejny obraz i aktualizuje scenę."""
        if self.indeks_obecnego_obrazu < len(self.lista_obrazow):
            sciezka_obrazu = os.path.join(self.folder_z_obrazami, self.lista_obrazow[self.indeks_obecnego_obrazu])
            logging.debug(f"Ładowanie obrazu: {sciezka_obrazu}")

            self.obraz = QImage(sciezka_obrazu)
            if self.obraz.isNull():
                QMessageBox.warning(self, "Błąd", f"Nie można wczytać obrazu: {sciezka_obrazu}")
                self.indeks_obecnego_obrazu += 1
                self.zaladuj_obraz()
                return

            pixmapa = QPixmap.fromImage(self.obraz)
            width = pixmapa.width()
            height = pixmapa.height()

            self.scena.clear()
            self.scena.setSceneRect(0, 0, width, height)
            self.scena.addPixmap(pixmapa)

            self.slider_x1.setMaximum(width)
            self.slider_x2.setMaximum(width)
            self.slider_y1.setMaximum(height)
            self.slider_y2.setMaximum(height)

            self.widok.setScene(self.scena)
            self.widok.fitInView(self.scena.sceneRect(), Qt.KeepAspectRatio)

            self.aktualizuj_licznik_obrazow()
        else:
            QMessageBox.information(self, "Koniec", "Wszystkie obrazy zostały przetworzone.")
            self.close()

    def aktualizuj_licznik_obrazow(self):
        """Aktualizuje etykietę z licznikiem obrazów."""
        self.etykieta_licznika_obrazow.setText(
            f"Obraz: {self.indeks_obecnego_obrazu + 1} / {self.liczba_obrazow}"
        )

    def aktualizuj_prostokat(self):
        """
        Aktualizuje wyświetlanie prostokąta na obrazie
        na podstawie aktualnych wartości suwaków.
        """
        if self.slider_y1.value() > self.slider_y2.value():
            self.slider_y2.setValue(self.slider_y1.value())
        if self.slider_x1.value() > self.slider_x2.value():
            self.slider_x2.setValue(self.slider_x1.value())

        x1 = self.slider_x1.value()
        x2 = self.slider_x2.value()
        y1 = self.slider_y1.value()
        y2 = self.slider_y2.value()

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        szerokosc = x2 - x1
        wysokosc = y2 - y1

        self.scena.clear()
        self.scena.addPixmap(QPixmap.fromImage(self.obraz))

        if szerokosc > 0 and wysokosc > 0:
            etykieta = Etykieta(x1, y1, szerokosc, wysokosc, self.kolory[self.indeks_obecnego_koloru])
            self.scena.addItem(etykieta)

    def dodaj_etykieta(self):
        """Dodaje etykietę (zapisuje współrzędne i klasę)."""
        klasa = self.wybor_klasy.currentText()
        x1 = self.slider_x1.value()
        x2 = self.slider_x2.value()
        y1 = self.slider_y1.value()
        y2 = self.slider_y2.value()

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        szerokosc = x2 - x1
        wysokosc = y2 - y1

        if szerokosc <= 0 or wysokosc <= 0:
            QMessageBox.warning(self, "Ostrzeżenie", "Nie można dodać etykiety o zerowych wymiarach.")
            return

        etykieta = Etykieta(x1, y1, szerokosc, wysokosc, self.kolory[self.indeks_obecnego_koloru])
        self.scena.addItem(etykieta)

        self.etykiety.append({
            'klasa': klasa,
            'prostokat': [x1, y1, szerokosc, wysokosc]
        })
        self.indeks_obecnego_koloru = (self.indeks_obecnego_koloru + 1) % len(self.kolory)

        self.slider_x1.setValue(0)
        self.slider_x2.setValue(0)
        self.slider_y1.setValue(0)
        self.slider_y2.setValue(0)

    def resetuj_etykiety(self):
        """Czyści listę etykiet i odświeża obraz."""
        self.etykiety.clear()
        self.scena.clear()
        self.zaladuj_obraz()

    def zapisz_adnotacje(self):
        """
        Zapisuje aktualne etykiety do pliku JSON i ładuje kolejny obraz.
        """
        self.dodaj_etykieta()

        adnotacje = []
        for etykieta in self.etykiety:
            klasa = etykieta['klasa']
            x_min, y_min, szerokosc, wysokosc = etykieta['prostokat']
            adnotacje.append({
                'class': klasa,
                'bbox': [x_min, y_min, szerokosc, wysokosc]
            })

        dane_obrazu = {
            'image': self.lista_obrazow[self.indeks_obecnego_obrazu],
            'annotations': adnotacje,
            'folder': self.folder_z_obrazami
        }

        self.istniejace_dane.append(dane_obrazu)

        save_annotations(self.plik_adnotacji, self.istniejace_dane)

        self.indeks_obecnego_obrazu += 1
        self.etykiety.clear()
        self.zaladuj_obraz()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AplikacjaOznaczaniaObrazow()
    ex.show()
    sys.exit(app.exec_())
