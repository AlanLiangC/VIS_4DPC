import os
import copy
import numpy as np
from pathlib import Path
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget


class ImageWindow(QMainWindow):

    def __init__(self) -> None:

        super(ImageWindow, self).__init__()

        self.monitor = QDesktopWidget().screenGeometry(1)
        self.monitor.setHeight(int(0.4 * self.monitor.height()))
        self.monitor.setWidth(int(0.9 * self.monitor.width()))

        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)
        self.layout = QGridLayout()

        self.init_window()

    def init_window(self):
        self.centerWidget.setLayout(self.layout)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label, 0, 0, 1, 3)
        self.width, self.height = self.image_label.width(), self.image_label.height()

    def display_image(self, image_array):
        # Ensure the input image is in the format of uint8
        assert image_array.dtype == np.uint8, "Image array must be of type np.uint8"

        # Convert the numpy array to QImage
        height, width, channels = image_array.shape
        bytes_per_line = width * channels
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap and display it
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)