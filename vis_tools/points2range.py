import cv2
import os.path as osp
import numpy as np
from . import dataset_parameters
from .points2pano import LiDAR_2_Pano

def create_rangeview(lidar: np.ndarray, save_path:str, dataname = 'kitti_od'):

    pano = LiDAR_2_Pano(lidar, dataset_parameters[dataname]['vis']['rangeview'])
    pano_save_path = save_path+'.npy'
    np.save(pano_save_path, pano)

    pano = (pano * 255).astype(np.uint8)
    cv2.imwrite(
        str(save_path+'.png'),
        cv2.applyColorMap(pano, 20),
    )
    pano = cv2.applyColorMap(pano, 20)
    
    cv2.imwrite(
    str('./rangeview.png'),
    pano,
    )
    return pano
