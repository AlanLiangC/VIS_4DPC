import os
import copy
import numpy as np
from pathlib import Path
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget

class ImageWindow(QMainWindow):
    def __init__(self, image_array):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.display_image(image_array)

    def display_image(self, image_array):
        # Ensure the input image is in the format of uint8
        assert image_array.dtype == np.uint8, "Image array must be of type np.uint8"

        # Convert the numpy array to QImage
        height, width, channels = image_array.shape
        bytes_per_line = channels * width
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap and display it
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    
    # Create a sample image as a numpy array (for testing purposes)
    sample_image = np.zeros((600, 800, 3), dtype=np.uint8)
    sample_image[100:500, 200:600] = [255, 0, 0]  # Draw a red rectangle
    
    window = ImageWindow(sample_image)
    window.show()
    sys.exit(app.exec_())
