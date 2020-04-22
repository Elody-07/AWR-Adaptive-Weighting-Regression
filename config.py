class Config(object):
    gpu_id = 1
    phase = 'train' 
    root_dir = '/home/ljs/pfren/dataset/'
    dataset = 'nyu' # ['nyu', 'icvl', 'msra', 'hands17']
    output_dir = './output/'
    load_model = ''

    test_id = 8 # for msra
    joint_num = JOINT[dataset]
    cube_size = [200, 200, 200]
    center_type = 'refine'  # ['joint_mean', 'joint', 'refine', 'mass', 'random']
    augment_para = [10, 0.1, 180] 

    input_size = 128
    downsample = 2 # [1,2,4]
    batch_size = 32
    max_epoch = 80
    loss_type = 'MyL1Loss'  # ['L1loss', 'mseloss', 'weighted_L1loss']
    coord_weight = 1000.0
    deconv_weight = 1.0
    kernel_size = 1   # for offset
    lr = 0.001

    print_freq = 100

    net = 'resnet_50'

JOINT = {
    'nyu': 14,
    'icvl': 16,
    'msra': 21,
    'hands17': 21,
    'itop': 15
}