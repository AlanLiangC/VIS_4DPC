from . import dvr
import torch
import torch.nn as nn
import numpy as np

from .generate_pure_lidar_points import LiDAR_Hardware_parameter, generate_custom_style_point_cloud

# os.environ['CUDA_VISIBLE_DEVICES']=f"{0}"
# torch.cuda.set_device(0)

def get_rendered_pcds(origin, points, tindex, gt_dist, pred_dist, pc_range,):
    pcds = []
    for t in range(len(origin)):
        mask = np.logical_and(tindex == t, gt_dist > 0.0)
        # skip the ones with no data
        if not mask.any():
            continue
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
        pcds.append(torch.from_numpy(pred_pts))
    return pcds

class Render_Points(nn.Module):
    '''
    Only for nuscenes to other format
    '''
    def __init__(self, data_config) -> None:
        super(Render_Points, self).__init__()

        pc_range = data_config['pc_range']
        self.pc_range = pc_range
        voxel_size = data_config['voxel_size']
        self.voxel_size = voxel_size

        self.n_height = int((pc_range[5] - pc_range[2]) / voxel_size)
        self.n_length = int((pc_range[4] - pc_range[1]) / voxel_size)
        self.n_width = int((pc_range[3] - pc_range[0]) / voxel_size)
        self.output_grid = [1, self.n_height, self.n_length, self.n_width]

        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor(pc_range[:3])[None, None, :], requires_grad=False
        )
        self.scaler = torch.nn.parameter.Parameter(
            torch.Tensor([voxel_size] * 3)[None, None, :], requires_grad=False
        )

    def numpy2tensor(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).cuda()
        return data

    def render_points(self, sigma, output_origin, target_dataname=None, target_points=None):

        res_output_origin = self.numpy2tensor(output_origin).view(1,1,-1)
        if target_points is None:
            assert target_dataname is not None
            target_points = generate_custom_style_point_cloud(**LiDAR_Hardware_parameter[target_dataname])
        res_output_points = self.numpy2tensor(target_points).unsqueeze(dim=0)

        output_origin, output_points, output_tindex = self.format_points(output_origin, target_points)

        pred_dist, gt_dist = dvr.render_forward(
            sigma, output_origin, output_points, output_tindex, self.output_grid, "test")
        
        pred_pcds = get_rendered_pcds(
                res_output_origin[0].cpu().numpy(),
                res_output_points[0].cpu().numpy(),
                output_tindex[0].cpu().numpy(),
                (gt_dist[0]*self.voxel_size).cpu().numpy(),
                (pred_dist[0]*self.voxel_size).cpu().numpy(),
                self.pc_range,
            )
        
        return pred_pcds[0]

    def format_points(self, output_origin, output_points):

        output_origin = self.numpy2tensor(output_origin).view(1,1,-1)
        output_points = self.numpy2tensor(output_points).unsqueeze(dim=0)

        output_origin = ((output_origin - self.offset) / self.scaler).float()
        output_points = ((output_points - self.offset) / self.scaler).float()
        output_tindex = output_points.new_zeros([1, output_points.shape[1]])

        return output_origin, output_points, output_tindex
    
    def convert_nuscenes(self, occ, output_origin, target_dataname=None, target_points=None):
        '''
        target_dataname: kitti-od / ArgoVerse
        '''

        occ = self.numpy2tensor(occ).permute(2,1,0).unsqueeze(dim=0).unsqueeze(dim=0)
        sigma = -torch.log(1-occ)

        rendered_points = self.render_points(sigma.contiguous(), output_origin, target_dataname, target_points)

        return rendered_points
    
    def expand_nuscenes_lines(self, point_cloud, num_beams_target):
        vertical_fov_min = -30
        vertical_fov_max = 10
        vertical_angles_original = np.linspace(np.radians(vertical_fov_min), np.radians(vertical_fov_max), 32)
        num_beam_times = num_beams_target // 32
        
        point_cloud_dict = {angle: [] for angle in vertical_angles_original}
        for point in point_cloud:
            elevation = np.arctan2(point[2], np.sqrt(point[0]**2 + point[1]**2))
            closest_angle = min(vertical_angles_original, key=lambda x: abs(x - elevation))
            point_cloud_dict[closest_angle].append(point)
        
        def interpolate_points(points, angles_original, num_beam_times):
            pcl_new = []
            for i in range(len(angles_original) - 1):
                angle1, angle2 = angles_original[i], angles_original[i + 1]
                points1, points2 = points[angle1], points[angle2]
                if points1 and points2:
                    for point1, point2 in zip(points1, points2):
                        new_points = [(1 - t) * np.array(point1) + t * np.array(point2) for t in np.linspace(0, 1, (num_beam_times+1))]
                        pcl_new.extend(new_points)
            return pcl_new

        target_points = interpolate_points(point_cloud_dict, vertical_angles_original, num_beam_times)
        return target_points

    def expand_to_other_lines(self, point_cloud, occ, output_origin, num_beams_target = 64):

        norm_point_cloud = point_cloud - output_origin.reshape(1, -1)
        target_points = self.expand_nuscenes_lines(norm_point_cloud, num_beams_target)
        target_points = target_points + output_origin.reshape(1, -1)

        occ = self.numpy2tensor(occ).permute(2,1,0).unsqueeze(dim=0).unsqueeze(dim=0)
        sigma = -torch.log(1-occ)

        rendered_points = self.render_points(sigma.contiguous(), output_origin, target_dataname=None, target_points = target_points)

        return rendered_points
    
    def change_lidar_fov(self, point_cloud, factor):
        '''
        factor: (0~1)
        '''
        pcl_new = []
        for point in point_cloud:
            distance = np.linalg.norm(point)
            elevation = np.arctan2(point[2], np.sqrt(point[0]**2 + point[1]**2))
            azimuth = np.arctan2(point[1], point[0])

            new_x = distance * np.cos(elevation*factor) * np.cos(azimuth)
            new_y = distance * np.cos(elevation*factor) * np.sin(azimuth)
            new_z = distance * np.sin(elevation*factor)
            pcl_new.append([new_x, new_y, new_z])

        return np.array(pcl_new)

    def change_lidar_state(self, point_cloud, occ, output_origin, state_dict, change_mode):
        '''
        change_mode: position, fov
        '''
        norm_point_cloud = point_cloud - output_origin.reshape(1, -1)
        position = state_dict['position']
        factor = state_dict['factor']

        if change_mode == 'position':
            output_origin = output_origin + position
            target_points = norm_point_cloud + output_origin.reshape(1, -1)
        elif change_mode == 'fov':
            target_points = self.change_lidar_fov(norm_point_cloud, factor)
            output_origin = output_origin + position
            target_points = target_points + output_origin.reshape(1, -1)
        else:
            raise NotImplementedError('No such mode!')

        occ = self.numpy2tensor(occ).permute(2,1,0).unsqueeze(dim=0).unsqueeze(dim=0)
        sigma = -torch.log(1-occ)

        rendered_points = self.render_points(sigma.contiguous(), output_origin, target_dataname=None, target_points = target_points)

        return rendered_points