import os
import copy
import numpy as np
from pathlib import Path
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget


class LiDAR_Control_Window(QMainWindow):

    def __init__(self, main_window) -> None:

        super(LiDAR_Control_Window, self).__init__()

        self.state_dict = dict(
            position = np.zeros(3),
            factor = 1
        )
        self.change_mode = None
        self.main_window = main_window
        self.monitor = QDesktopWidget().screenGeometry(1)
        self.monitor.setHeight(int(0.3 * self.monitor.height()))
        self.monitor.setWidth(int(0.3 * self.monitor.width()))

        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)
        self.layout = QGridLayout()

        self.init_window()

    def init_window(self):
        self.centerWidget.setLayout(self.layout)

        # Buttons
        # up y axis
        self.up_y_btn = QPushButton("+y")
        self.layout.addWidget(self.up_y_btn, 0, 1, 1, 1)
        self.up_y_btn.pressed.connect(self.up_y)

        # down y axis
        self.down_y_btn = QPushButton("-y")
        self.layout.addWidget(self.down_y_btn, 2, 1, 1, 1)
        self.down_y_btn.pressed.connect(self.down_y)

        # up x axis
        self.up_x_btn = QPushButton("+x")
        self.layout.addWidget(self.up_x_btn, 1, 2, 1, 1)
        self.up_x_btn.pressed.connect(self.up_x)

        # down x axis
        self.down_x_btn = QPushButton("-x")
        self.layout.addWidget(self.down_x_btn, 1, 0, 1, 1)
        self.down_x_btn.pressed.connect(self.down_x)

        # up z axis
        self.up_z_btn = QPushButton("+z")
        self.layout.addWidget(self.up_z_btn, 0, 0, 1, 1)
        self.up_z_btn.pressed.connect(self.up_z)

        # down z axis
        self.down_z_btn = QPushButton("-x")
        self.layout.addWidget(self.down_z_btn, 0, 2, 1, 1)
        self.down_z_btn.pressed.connect(self.down_z)

        # up fov
        self.up_fov_btn = QPushButton("+fov")
        self.layout.addWidget(self.up_fov_btn, 2, 2, 1, 1)
        self.up_fov_btn.pressed.connect(self.up_fov)

        # down fov
        self.down_fov_btn = QPushButton("-fov")
        self.layout.addWidget(self.down_fov_btn, 2, 0, 1, 1)
        self.down_fov_btn.pressed.connect(self.down_fov)

    def up_y(self):
        self.change_mode = 'position'
        self.state_dict['position'][1] += 10
        print(self.state_dict)
        self.show_main_window()
    def down_y(self):
        self.change_mode = 'position'
        self.state_dict['position'][1] -= 1
        self.show_main_window()
    def up_x(self):
        self.change_mode = 'position'
        self.state_dict['position'][0] += 1
        self.show_main_window()
    def down_x(self):
        self.change_mode = 'position'
        self.state_dict['position'][0] -= 1
        self.show_main_window()
    def up_z(self):
        self.change_mode = 'position'
        self.state_dict['position'][2] += 1
        self.show_main_window()
    def down_z(self):
        self.change_mode = 'position'
        self.state_dict['position'][2] -= 1
        self.show_main_window()
    def up_fov(self):
        self.change_mode = 'fov'
        self.state_dict['factor'] += 0.1
        self.show_main_window()
    def down_fov(self):
        self.change_mode = 'fov'
        self.state_dict['factor'] -= 0.1
        self.show_main_window()

    def show_main_window(self):
        
        self.main_window.show_changed_lidar(self.state_dict, self.change_mode)