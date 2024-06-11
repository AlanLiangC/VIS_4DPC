from torch.utils.cpp_extension import load

dvr = load("dvr", sources=["ops/dvr/dvr.cpp", "ops/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])

dataset_parameters = {
    'nusc': {
        '1s' : {
            'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            'encoding_voxel_size': [0.1, 0.1, 0.2],
            'voxel_size': 0.2,
            'n_input': 2,
            'input_step': 1,
            'n_output': 2,
            'input_step': 1,
            'metrics': 'l1'},

        '3s': {
            'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            'encoding_voxel_size': [0.1, 0.1, 0.2],
            'voxel_size': 0.2,
            'n_input': 6,
            'input_step': 1,
            'n_output': 6,
            'input_step': 1,
            'metrics': 'l1'},

        'vis': {
            'rangeview':{
                'lidar_H': 34,
                'lidar_W': 515,
                'intrinsics': (10.0, 40.0)
            }
        }
},

    'kitti_od': {
        '1s' : {
            'pc_range': [-51.2, -51.2, -4.0, 51.2, 51.2, 4.0],
            'encoding_voxel_size': [0.1, 0.1, 0.2],
            'voxel_size': 0.2,
            'n_input': 5,
            'input_step': 2,
            'n_output': 5,
            'output_step': 2,
            'metrics': 'l1'},

        '3s': {
            'pc_range': [-51.2, -51.2, -4.0, 51.2, 51.2, 4.0],
            'encoding_voxel_size': [0.1, 0.1, 0.2],
            'voxel_size': 0.2,
            'n_input': 5,
            'input_step': 6,
            'n_output': 5,
            'output_step': 6,
            'metrics': 'l1'},

        'vis': {
            'rangeview':{
                'lidar_H': 66,
                'lidar_W': 1030,
                'intrinsics': (2.0, 26.9)
            }
        }
    }
}
