# -----------------------------------------
# python modules
# -----------------------------------------
from importlib import import_module
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import os

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.imdb_util import *

test_list = [
# 'kitti_3d_multi_quan2x2_resnet50',
# 'kitti_3d_multi_quan4x4_resnet50',
# 'kitti_3d_multi_quan8x8_resnet50',
# 'kitti_3d_multi_quan16x16_resnet50',
# 'kitti_3d_multi_quan8x8_resnet50_lr',
'kitti_3d_multi_main_resnet50',]
model_weight_list = [
    # 'model_10000_pkl',
    # 'model_20000_pkl',
    # 'model_30000_pkl',
    # 'model_40000_pkl',
    # 'model_50000_pkl',
    # 'model_60000_pkl',
    # 'model_70000_pkl',
    # 'model_80000_pkl',
    # 'model_90000_pkl',
    'model_100000_pkl',
]
for i in range(len(test_list)):
    test_file = test_list[i]
    conf_path = f'/media/hdd/jhkim/git/M3D-RPN/output/{test_file}/conf.pkl'
    for j in range(len(model_weight_list)):
        model_weight = model_weight_list[j]
        weights_path = f'/media/hdd/jhkim/git/M3D-RPN/output/{test_file}/weights/{model_weight}'

        # load config
        conf = edict(pickle_read(conf_path))
        conf.pretrained = None
        conf['model_save_name'] = test_file
        print(conf['model_save_name'])
        conf['model_weight'] = model_weight
        data_path = os.path.join(os.getcwd(), 'data')
        results_path = os.path.join('output', test_file, 'results', model_weight, 'data')
        results_save_path = os.path.join('output', 'test_time')

        # make directory
        mkdir_if_missing(results_path, delete_if_exist=False)
        mkdir_if_missing(results_save_path, delete_if_exist=False)
        
        # -----------------------------------------
        # torch defaults
        # -----------------------------------------

        # defaults
        init_torch(conf.rng_seed, conf.cuda_seed)

        # -----------------------------------------
        # setup network
        # -----------------------------------------

        # net
        net = import_module('models.' + conf.model).build(conf)

        # load weights
        load_weights(net, weights_path, remove_module=True)

        # switch modes for evaluation
        net.eval()

        print(pretty_print('conf', conf))

        # -----------------------------------------
        # test kitti
        # -----------------------------------------

        test_kitti_3d_q_for_inference_time(conf.dataset_test, net, conf, results_path, data_path, results_save_path, use_log=False)