import sys
import numpy as np
from PyQt6.QtWidgets import (QWidget, QApplication, QPushButton, QMenuBar, QMenu,
                             QHBoxLayout, QVBoxLayout, QLabel, QMainWindow,
                             QFileDialog, QTableWidget, QTableWidgetItem)

from PyQt6.QtGui import QPainter, QAction

from PyQt6.QtCore import Qt, pyqtSignal
from PIL import Image
import torch
from torchvision import transforms
from model import Model
import torch.nn.functional as F

# Load trained model
model = Model()
model.load_state_dict(torch.load("../model.pth", map_location="cpu"))
model.eval()

def predict_digit(canvas_widget: QWidget, label: QLabel = None):
    img = canvas_widget.get_image().astype(np.float32) / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081])
    ])
    img = transform(img).unsqueeze(0)

    with torch.inference_mode():
        output = model(img)
        probs = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        pred_text = f"Predicted Digit: {pred}"
        print(pred_text)

        if label:
            label.setText(pred_text)

        return probs.squeeze().tolist()

# --- Canvas Widget ---
class CanvasWidget(QWidget):
    drawSignal = pyqtSignal()
    def __init__(self, grid_size=28, pixel_size=10):
        super().__init__()
        self.grid_size = grid_size
        self.pixel_size = pixel_size
        self.setFixedSize(grid_size * pixel_size, grid_size * pixel_size)
        self.canvas = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.text_color = Qt.GlobalColor.black
        self.bg_color = Qt.GlobalColor.white


    def paintEvent(self, event):
        painter = QPainter(self)

        painter.fillRect(self.rect(), self.bg_color)

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.canvas[y, x] > 0:
                    painter.fillRect(
                        x * self.pixel_size,
                        y * self.pixel_size,
                        self.pixel_size,
                        self.pixel_size,
                        self.text_color,
                    )

    def mousePressEvent(self, event):
        self._draw(event.pos())

        self.drawSignal.emit()

    def mouseMoveEvent(self, event):
        self._draw(event.pos())

        self.drawSignal.emit()

    def _draw(self, pos):
        x = pos.x() // self.pixel_size
        y = pos.y() // self.pixel_size
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.canvas[y, x] = 255
            self.update()

    def clear(self):
        self.canvas.fill(0)
        self.update()

    def get_image(self):
        return self.canvas.copy()

    def save(self, filename):
        img = Image.fromarray(self.canvas).convert("L")
        img = img.resize((280, 280), Image.NEAREST)
        img.save(filename)
        print(f"Saved to {filename}")

class ProbabilitiesWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setRowCount(10)
        self.table.setHorizontalHeaderLabels(["Digit", "Probability"])
        self.layout.addWidget(self.table)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.setLayout(self.layout)

        tableFont = self.table.font()
        tableFont.setPixelSize(20)
        self.table.setFont(tableFont)

    def populate(self, sorted_probs):
        for (i, s) in enumerate(sorted_probs):
            self.table.setItem(i, 0, QTableWidgetItem(f"{s[0]:.2f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{s[1]:.2f}"))

# --- Main App Widget ---
class DrawWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("28x28 Drawing Pad")
        #self.setFixedSize(280, 330)
        self.darkMode = False
        self.liveProbabilityMode = True

        self.menubar = QMenuBar()

        self.setMenuBar(self.menubar)

        self.fileMenu = QMenu("File")

        self.fileMenu__save = QAction("Save")
        self.fileMenu.addAction(self.fileMenu__save)

        self.fileMenu__save.triggered.connect(self.saveAction)

        self.editMenu = QMenu("Edit")

        self.editMenu__darkMode = QAction("Dark Mode")

        self.editMenu__darkMode.setCheckable(True)
        self.editMenu__darkMode.setChecked(self.darkMode)

        self.editMenu__liveProbabilityMode = QAction("Live Probability Mode")
        self.editMenu__liveProbabilityMode.setCheckable(True)
        self.editMenu__liveProbabilityMode.setChecked(self.liveProbabilityMode)

        self.editMenu.addAction(self.editMenu__darkMode)
        self.editMenu__darkMode.triggered.connect(self.toggle_dark_mode)

        self.editMenu.addAction(self.editMenu__liveProbabilityMode)
        self.editMenu__liveProbabilityMode.triggered.connect(self.toggle_live_probability_mode)

        self.menubar.addMenu(self.fileMenu)
        self.menubar.addMenu(self.editMenu)

        # Canvas as its own widget
        self.canvas_widget = CanvasWidget()

        if self.liveProbabilityMode:
            self.canvas_widget.drawSignal.connect(self.predict)

        # Buttons
        self.clear_btn = QPushButton("Clear")
        self.predict_btn = QPushButton("Predict")

        self.clear_btn.clicked.connect(self.clear_btn_clicked)

        #self.save_btn.clicked.connect(lambda: self.canvas_widget.save("../digit.jpg"))
        self.predict_btn.clicked.connect(self.predict)

        self.pred_widget_label = QLabel("Predicted Digit: ")

        self.setContentsMargins(0, 0, 0, 0)

        pred_widget_font = self.pred_widget_label.font()
        pred_widget_font.setPixelSize(20)
        self.pred_widget_label.setFont(pred_widget_font)

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.predict_btn)

        canvas_layout = QHBoxLayout()
        canvas_layout.addStretch()
        canvas_layout.addWidget(self.canvas_widget)
        canvas_layout.addStretch()

        # This is the layout for the init canvas
        main_1_layout = QVBoxLayout()
        main_1_layout.addWidget(self.pred_widget_label)
        main_1_layout.addLayout(canvas_layout)
        main_1_layout.addLayout(button_layout)

        main_1_layout.setContentsMargins(0, 0, 0, 0)

        self.main_2_layout = QVBoxLayout() # This is the layout for the detailed probability view

        self.probabilities_widget = ProbabilitiesWidget()
        self.probabilities_widget.setVisible(self.liveProbabilityMode)
        self.main_2_layout.addWidget(self.probabilities_widget)

        main_layout = QHBoxLayout()
        main_layout.addLayout(main_1_layout)
        main_layout.addLayout(self.main_2_layout)

        mainWidget = QWidget()
        mainWidget.setLayout(main_layout)

        self.setCentralWidget(mainWidget)

    def predict(self):
        Y = predict_digit(self.canvas_widget, self.pred_widget_label)

        digit_probs = list(enumerate(Y))
        sorted_probs = sorted(digit_probs, key = lambda x: x[1], reverse=True)
        self.probabilities_widget.populate(sorted_probs)

    def saveAction(self):
        filename, = QFileDialog.getSaveFileName()
        if filename != "":
            self.canvas_widget.save(filename)

    def clear_btn_clicked(self):
        self.pred_widget_label.setText("Predicted Digit: ")
        self.canvas_widget.clear()
        self.probabilities_widget.table.clearContents()

    def toggle_dark_mode(self):
        if self.darkMode:
            self.canvas_widget.bg_color = Qt.GlobalColor.white
            self.canvas_widget.text_color = Qt.GlobalColor.black
        else:
            self.canvas_widget.bg_color = Qt.GlobalColor.black
            self.canvas_widget.text_color = Qt.GlobalColor.white

        self.darkMode = not self.darkMode
        self.canvas_widget.update()

    def toggle_live_probability_mode(self):
        self.liveProbabilityMode = not self.liveProbabilityMode
        self.probabilities_widget.setVisible(self.liveProbabilityMode)
        if self.liveProbabilityMode:
            self.canvas_widget.drawSignal.connect(self.predict)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrawWidget()
    window.show()
    sys.exit(app.exec())
