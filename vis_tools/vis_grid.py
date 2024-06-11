offscreen = False

from mayavi import mlab
mlab.options.offscreen = offscreen
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import time, os.path as osp, numpy as np
import matplotlib; matplotlib.use('agg')

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid


def draw_occ(
    voxels,          # semantic occupancy predictions
    pred_pts,        # lidarseg predictions
    vox_origin,
    voxel_size=0.2,  # voxel size in the real world
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    mode=0,
    sem=False,
    dataset='nuscenes',
    save_path=None
):
    w, h, z = voxels.shape
    # grid = grid.astype(np.int)

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    if mode == 0:
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
        
    elif mode == 1:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        pred_pts = pred_pts[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, pred_pts.reshape(-1)]).T
    elif mode == 2:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        gt_label = pt_label[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, gt_label.reshape(-1)]).T
    else:
        raise NotImplementedError

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 100)
    ]
    print(len(fov_voxels))
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    if not sem:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="jet",
            # colormap="hot",
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
        )
    else:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=1,
            vmax=19 if dataset == 'kitti' else 16, # 16
        )

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"

    scene = figure.scene

    # camera view
    scene.camera.position = np.array([0, 0, 0.]) # - np.array([0.7, 1.3, 0.])
    scene.camera.focal_point = np.array([0.7, 1.3, 0.]) # - np.array([0.7, 1.3, 0.])
    scene.camera.view_angle = 41
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [0.01, 300.]
    scene.camera.compute_view_plane_normal()
    scene.render()

    # mlab.show()

    # mlab.close()

if __name__ == "__main__":
    # KITTI
    voxel = np.load('Vis_Semantic_KITTI/temp/0.npy')
    voxel[:,:,-10:] = 0
    point_cloud_range = [-70, -70 -4.5, 70, 70, 4.5]

    # Nuscenes
    # voxel = np.load('Vis_Semantic_KITTI/temp/labels.npz')['semantics']
    # voxel = np.where(voxel > 16, 0, 1)
    # point_cloud_range = [-40, -40, -3.4, 40, 40, 3.0]

    voxel_origin = point_cloud_range[:3]
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    grid_size = voxel.shape
    for z in range(grid_size[2]):
        mask = (voxel > 0)[..., z]
        voxel[..., z][mask] = z + 1 # grid_size[2] - z

    draw_occ(
        voxels=voxel,
        pred_pts=None,
        vox_origin=voxel_origin,
        voxel_size=[0.2]*3,
        grid=None, # grid.squeeze(0).cpu().numpy(), 
        pt_label=None, # pt_label.squeeze(-1),
        save_dir=None,
        cam_positions=None,
        focal_positions=None,
        timestamp=timestamp,
        mode=0,
        sem=False,
        dataset='kitti',
        save_path='Vis_Semantic_KITTI/temp/'
    )
