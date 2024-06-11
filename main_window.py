import sys
sys.path.append('/home/alan/AlanLiang/Projects/3D_Reconstruction/AlanLiang/VIS_4DPC')
import os
import os.path as osp
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QMainWindow, QDesktopWidget, \
            QGridLayout, QPushButton, QFileDialog, QComboBox
import pyqtgraph.opengl as gl

import torch
import numpy as np
from utils import common
from vis_tools.points2range import create_rangeview
from vis_tools.vis_grid import draw_occ
from vis_tools import dataset_parameters
from vis_tools.render_points import Render_Points
from windows.image_window import ImageWindow
from windows.lidar_control_window  import LiDAR_Control_Window

os.environ['CUDA_VISIBLE_DEVICES']=f"{0}"
torch.cuda.set_device(0)

class FDPCFWindow(QMainWindow):

    def __init__(self) -> None:
        super(FDPCFWindow, self).__init__()

        self.succecc_show = False
        self.image_window = ImageWindow()
        self.monitor = QDesktopWidget().screenGeometry(1)
        self.monitor.setHeight(int(self.monitor.height() * 0.8))
        self.monitor.setWidth(int(self.monitor.width() * 0.8))

        self.grid_dimensions = 20
        self.index = 0

        data_config = dataset_parameters['nusc']['3s']
        self.render_model = Render_Points(data_config).cuda()

        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)
        self.layout = QGridLayout()
        self.viewer = common.AL_viewer()
        self.grid = gl.GLGridItem()
        self.init_window()

    def init_window(self):
        self.centerWidget.setLayout(self.layout)

        self.viewer.setWindowTitle('drag & drop point cloud viewer')
        self.viewer.setCameraPosition(distance=2 * self.grid_dimensions)
        self.layout.addWidget(self.viewer, 0, 0, 1, 5)

        # grid
        self.grid.setSize(self.grid_dimensions, self.grid_dimensions)
        self.grid.setSpacing(1, 1)
        self.grid.translate(0, 0, -2)

        self.viewer.addItem(self.grid)
        # Buttons
        # Load data
        self.select_vis_space_btn = QPushButton("Select VIS Space")
        self.layout.addWidget(self.select_vis_space_btn, 1, 0, 1, 1)
        self.select_vis_space_btn.pressed.connect(self.select_vis_space)

        # Show_points
        self.show_points_btn = QPushButton("Show Points")
        self.layout.addWidget(self.show_points_btn, 2, 0, 1, 1)
        self.show_points_btn.pressed.connect(self.show_points)

        # Show_range_view
        self.show_rangeview_btn = QPushButton("Show RangeView")
        self.layout.addWidget(self.show_rangeview_btn, 2, 1, 1, 1)
        self.show_rangeview_btn.pressed.connect(self.show_rangeview)
        self.show_rangeview_btn.setEnabled(False)

        # Show_occ
        self.show_occ_btn = QPushButton("Show Occ")
        self.layout.addWidget(self.show_occ_btn, 2, 2, 1, 1)
        self.show_occ_btn.pressed.connect(self.show_occ)
        self.show_occ_btn.setEnabled(False)

        # Save camera image
        self.save_viewer_image_btn = QPushButton("Save Viewer image")
        self.layout.addWidget(self.save_viewer_image_btn, 2, 3, 1, 1)
        self.save_viewer_image_btn.pressed.connect(self.save_viewer_image)
        self.save_viewer_image_btn.setEnabled(False)

        # Show diff occ
        self.show_diff_occ_btn = QPushButton("Show diff occ")
        self.layout.addWidget(self.show_diff_occ_btn, 2, 4, 1, 1)
        self.show_diff_occ_btn.pressed.connect(self.show_diff_occ)
        self.show_diff_occ_btn.setEnabled(False)

        # lidar control
        self.show_lidar_control_btn = QPushButton("Lidar Control")
        self.layout.addWidget(self.show_lidar_control_btn, 3, 2, 1, 1)
        self.show_lidar_control_btn.pressed.connect(self.show_lidar_control)
        self.show_lidar_control_btn.setEnabled(False)

    
        # update lidar control
        # self.update_lidar_control_btn = QPushButton("Update Lidar Control")
        # self.layout.addWidget(self.update_lidar_control_btn, 3, 3, 1, 1)
        # self.update_lidar_control_btn.pressed.connect(self.update_lidar_control)
        # self.update_lidar_control_btn.setEnabled(False)

        # QComboBox
        # model
        self.models_combo_box = QComboBox(self)
        self.layout.addWidget(self.models_combo_box, 1, 1, 1, 1)
        self.models_combo_box.currentIndexChanged.connect(self.swich_combox)

        # dir
        self.models_dir_combo_box = QComboBox(self)
        self.layout.addWidget(self.models_dir_combo_box, 1, 2, 1, 1)
        for fold in ['input', 'gt_output', 'pred_output']:
            self.models_dir_combo_box.addItem(fold)
        self.models_dir_combo_box.currentIndexChanged.connect(self.swich_combox)

        # batch
        self.batch_combo_box = QComboBox(self)
        self.layout.addWidget(self.batch_combo_box, 1, 3, 1, 1)
        self.batch_combo_box.currentIndexChanged.connect(self.swich_combox)

        # timestamp
        self.timestamp_combo_box = QComboBox(self)
        self.layout.addWidget(self.timestamp_combo_box, 1, 4, 1, 1)
        self.timestamp_combo_box.currentIndexChanged.connect(self.swich_combox)

        # change style
        self.change_style_combo_box = QComboBox(self)
        self.layout.addWidget(self.change_style_combo_box, 3, 0, 1, 1)
        for style in ['', 'kitti-od', 'ArgoVerse']:
            self.change_style_combo_box.addItem(style)
        self.change_style_combo_box.currentIndexChanged.connect(self.swich_change_style_combox)
        self.change_style_combo_box.setEnabled(False)

        # expand line
        self.expand_line_combo_box = QComboBox(self)
        self.layout.addWidget(self.expand_line_combo_box, 3, 1, 1, 1)
        for style in ['', '64', '128', '256']:
            self.expand_line_combo_box.addItem(style)
        self.expand_line_combo_box.currentIndexChanged.connect(self.swich_expand_line_combox)
        self.expand_line_combo_box.setEnabled(False)

    def reset_viewer(self):

        self.viewer.items = []

    def reset_image_window(self):
        self.image_window.hide()

    def split_vis_space(self, vis_space:str):
        vis_space_split = vis_space.split('/')
        self.vis_info = dict(
            dataname = vis_space_split[-3],
            forecasting_time = vis_space_split[-2]
        )
        if vis_space_split[-3] == 'nusc':
            self.change_style_combo_box.setEnabled(True)
            self.expand_line_combo_box.setEnabled(True)

    def init_batch_timestamp(self):
        occ_path = osp.join(self.vis_space, 'ori', 'occ.npy')
        occ_data = np.load(occ_path)
        batch_size = occ_data.shape[0]
        timestamps_lenth = occ_data.shape[1]

        for i in range(batch_size) :
            self.batch_combo_box.addItem(str(i))

        for i in range(timestamps_lenth) :
            self.timestamp_combo_box.addItem(str(i))

    def select_vis_space(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "", options=options)
        if directory:
            self.vis_space = directory
            self.split_vis_space(directory)
            self.model_names = os.listdir(directory)
            for model_name in self.model_names:
                self.models_combo_box.addItem(model_name)

            assert 'ori' in self.model_names
            self.init_batch_timestamp()

        else:
            raise NotImplementedError('No such fold space!')

    @property
    def get_points_path(self):
        
        model_name = self.models_combo_box.currentText()
        sub_fold = self.models_dir_combo_box.currentText()
        batch_idx = self.batch_combo_box.currentText()
        timespamp_idx = self.timestamp_combo_box.currentText()
        return osp.join(self.vis_space, model_name, sub_fold, f'{batch_idx}_{timespamp_idx}.txt')
    
    @property
    def get_sample_path(self):
        
        model_name = self.models_combo_box.currentText()
        sub_fold = self.models_dir_combo_box.currentText()
        batch_idx = self.batch_combo_box.currentText()
        timespamp_idx = self.timestamp_combo_box.currentText()
        return osp.join(self.vis_space, model_name, sub_fold, f'{batch_idx}_{timespamp_idx}')

    @property
    def get_occ_path(self):

        model_name = self.models_combo_box.currentText()
        return osp.join(self.vis_space, model_name, 'occ.npy')

    @property
    def get_output_origin_path(self):

        model_name = self.models_combo_box.currentText()
        return osp.join(self.vis_space, model_name, 'output_origin.npy')

    def swich_combox(self):
        if self.succecc_show:
            self.show_points()

    def show_points(self):
        self.reset_viewer()

        points_path = self.get_points_path
        points = np.loadtxt(points_path)
        points = points[points[:,2] < 3]
        self.current_points = points

        mesh = common.get_points_mesh(points, 5)
        self.viewer.addItem(mesh)
        if not self.succecc_show:
            self.show_rangeview_btn.setEnabled(True)
            self.show_occ_btn.setEnabled(True)
            self.save_viewer_image_btn.setEnabled(True)
            self.show_diff_occ_btn.setEnabled(True)
            self.show_lidar_control_btn.setEnabled(True)
            # self.update_lidar_control_btn.setEnabled(True)
        self.succecc_show = True

    def show_custom_points(self, points, point_size=5):
        self.reset_viewer()
        mesh = common.get_points_mesh(points, point_size)
        self.viewer.addItem(mesh)

    def show_rangeview(self):
        pano = create_rangeview(self.current_points,
                         save_path=self.get_sample_path,
                         dataname=self.vis_info['dataname'])
        
        self.image_window.display_image(pano)
        self.image_window.show()

    def show_occ(self):
        batct_idx = int(self.batch_combo_box.currentText())
        timestamp = int(self.timestamp_combo_box.currentText())
        current_occ = np.load(self.get_occ_path)
        occ = current_occ[batct_idx,timestamp,...]
        grid_size = occ.shape
        occ = np.where(occ >= 0.1, occ, 0)
        for z in range(grid_size[2]):
            mask = (occ > 0)[..., z]
            occ[..., z][mask] = z + 1 # grid_size[2] - z

        # occ[...,-18:] = 0

        draw_occ(occ,
                 pred_pts=None,
                 vox_origin=dataset_parameters[self.vis_info['dataname']][self.vis_info['forecasting_time']]['pc_range'][:3],
                 voxel_size=[0.2]*3,
                 grid=None,
                 pt_label=None,
                 mode=0,
                 sem=False,
                 dataset=self.vis_info['dataname'],
                 save_path=self.get_sample_path
                 )

    def save_viewer_image(self):
        image = self.viewer.readQImage()
        image.save('./screenshot.png')

    def show_diff_occ(self):
        batct_idx = int(self.batch_combo_box.currentText())
        timestamp = int(self.timestamp_combo_box.currentText())
        ori_occ_path = osp.join(self.vis_space, 'ori', 'occ.npy')
        our_occ_path = osp.join(self.vis_space, 'lstm', 'occ.npy')

        ori_occ = np.load(ori_occ_path)
        our_occ = np.load(our_occ_path)

        occ = our_occ[batct_idx,timestamp,...] - ori_occ[batct_idx,timestamp,...]
        occ = np.where(occ >= 0.9, occ, 0)

        grid_size = occ.shape
        for z in range(grid_size[2]):
            mask = (occ > 0)[..., z]
            occ[..., z][mask] = z + 1 # grid_size[2] - z

        # occ[...,-10:] = 0

        draw_occ(occ,
                 pred_pts=None,
                 vox_origin=dataset_parameters[self.vis_info['dataname']][self.vis_info['forecasting_time']]['pc_range'][:3],
                 voxel_size=[0.2]*3,
                 grid=None,
                 pt_label=None,
                 mode=0,
                 sem=False,
                 dataset=self.vis_info['dataname'],
                 save_path=self.get_sample_path
                 )

    def swich_change_style_combox(self):

        target_dataname = self.change_style_combo_box.currentText()

        if target_dataname:

            batct_idx = int(self.batch_combo_box.currentText())
            timestamp = int(self.timestamp_combo_box.currentText())
            current_occ = np.load(self.get_occ_path)
            occ = current_occ[batct_idx,timestamp,...]
            output_origin = np.zeros([1,3])
            rendered_points = self.render_model.convert_nuscenes(occ, output_origin,target_dataname=target_dataname)
            self.show_custom_points(rendered_points)

        else:
            self.show_custom_points(self.current_points)

    def swich_expand_line_combox(self):
        beams_num = self.expand_line_combo_box.currentText()
        if beams_num:

            batct_idx = int(self.batch_combo_box.currentText())
            timestamp = int(self.timestamp_combo_box.currentText())
            current_occ = np.load(self.get_occ_path)
            current_output_origin = np.load(self.get_output_origin_path)
            occ = current_occ[batct_idx,timestamp,...]
            output_origin = current_output_origin[batct_idx,timestamp,...]
            rendered_points = self.render_model.expand_to_other_lines(self.current_points, occ, output_origin, num_beams_target=eval(beams_num))
            self.show_custom_points(rendered_points, point_size=2.5)

        else:
            self.show_custom_points(self.current_points)

    def init_lidar_control_window(self):
        if hasattr(self, 'lidar_cont_window'):
            pass
        else:
            self.lidar_cont_window = LiDAR_Control_Window(self)

    def show_lidar_control(self):
        self.init_lidar_control_window()
        self.lidar_cont_window.show()

    # def update_lidar_control(self):
    #     state_dict = self.lidar_cont_window.state_dict
    #     change_mode = self.lidar_cont_window.change_mode
    #     self.show_changed_lidar(state_dict, change_mode)

    def show_changed_lidar(self, state_dict, change_mode):

        state_dict = self.lidar_cont_window.state_dict
        change_mode = self.lidar_cont_window.change_mode
        batct_idx = int(self.batch_combo_box.currentText())
        timestamp = int(self.timestamp_combo_box.currentText())
        current_occ = np.load(self.get_occ_path)
        current_output_origin = np.load(self.get_output_origin_path)
        occ = current_occ[batct_idx,timestamp,...]
        output_origin = current_output_origin[batct_idx,timestamp,...]

        rendered_points = self.render_model.change_lidar_state(self.current_points, occ, output_origin, state_dict, change_mode)
        # self.current_points = rendered_points
        self.show_custom_points(rendered_points)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = FDPCFWindow()
    window.show()
    app.exec_()