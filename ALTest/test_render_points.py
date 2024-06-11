import os 
import sys
import torch
sys.path.append('/home/alan/AlanLiang/Projects/3D_Reconstruction/AlanLiang/VIS_4DPC')
import numpy as np
from vis_tools.render_points import Render_Points
from vis_tools import dataset_parameters

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']=f"{0}"
    torch.cuda.set_device(0)
    data_config = dataset_parameters['nusc']['3s']
    model = Render_Points(data_config).cuda()

    points = np.loadtxt('vis_results/nusc/3s/25/lstm/pred_output/0_0.txt')
    occ = np.load('vis_results/nusc/3s/25/lstm/occ.npy')[0,0,...]
    output_origin = np.zeros([1,3])
    rendered_points = model.convert_nuscenes(occ, output_origin,target_dataname='ArgoVerse')
    print(rendered_points.shape)
    np.savetxt('ALTest/temp/temp.txt', rendered_points)