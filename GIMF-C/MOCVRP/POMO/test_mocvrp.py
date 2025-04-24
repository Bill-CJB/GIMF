##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 6

##########################################################################################
# Path Config
import os
import sys
import torch
import numpy as np


os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src
from utils.cal_pareto_demo import Pareto_sols
from utils.cal_ps_hv import cal_ps_hv

from MOCVRPTester import CVRPTester as Tester
from MOCVRProblemDef import get_random_problems
# from generate_test_dataset import load_dataset
##########################################################################################
import time


from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
##########################################################################################
# parameters

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'in_channels': 1,
    'patch_size': 16,
    'pixel_density': 10,
    'fusion_layer_num': 3,
    'bn_num': 10,
    'bn_img_num': 10,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/train_cvrp',  # directory path of pre-trained model and log files saved.
        'epoch': 200,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 200,
    'test_batch_size': 200,
    'augmentation_enable': True,
    'aug_factor': 1,
    'aug_batch_size': 200
}

logger_params = {
    'log_file': {
        'desc': 'test__cvrp',
        'filename': 'run_log'
    }
}

##########################################################################################
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]
##########################################################################################
def main(n_sols = 101):

    timer_start = time.time()
    
    if DEBUG_MODE:
        _set_debug_mode()
    
    create_logger(**logger_params)
    _print_config()
    
    tester = Tester(model_params=model_params,
                    tester_params=tester_params)
    
    copy_all_src(tester.result_folder)


    problem_size = 20

    if tester_params['augmentation_enable']:
        test_name = 'Aug'
        tester_params['test_batch_size'] = tester_params['aug_batch_size']
    else:
        test_name = 'NoAug'

    test_path = f"./data/testdata_cvrp_size{problem_size}.pt"
    data = torch.load(test_path)
    shared_node_demand = data['demand_data'].squeeze(-1).to(device=CUDA_DEVICE_NUM)
    shared_depot_xy = data['node_data'][:, 0, :].unsqueeze(1).to(device=CUDA_DEVICE_NUM)
    shared_node_xy = data['node_data'][:, 1:, :].to(device=CUDA_DEVICE_NUM)

    batch_size = shared_node_xy.shape[0]
    sols = np.zeros([batch_size, n_sols, 2])
    for i in range(n_sols):
        pref = torch.zeros(2).cuda()
        pref[0] = 1 - 0.01 * i
        pref[1] = 0.01 * i
        pref = pref / torch.sum(pref)
        pref = pref[None, :].expand(shared_depot_xy.size(0), 2)
    
        aug_score = tester.run(shared_depot_xy, shared_node_xy, shared_node_demand, pref)
        sols[:, i, 0] = np.array(aug_score[0].flatten())
        sols[:, i, 1] = np.array(aug_score[1].flatten())
        
    timer_end = time.time()
    
    total_time = timer_end - timer_start


    if problem_size == 20:
        ref = np.array([30,4])    #20
    elif problem_size == 50:
        ref = np.array([45,4])   #50
    elif problem_size == 100:
        ref = np.array([80,4])   #100
    else:
        print('Have yet define a reference point for this problem size!')


    nd_sort = Pareto_sols(p_size=problem_size, pop_size=sols.shape[0], obj_num=sols.shape[2])
    sols_t = torch.Tensor(sols)
    nd_sort.update_PE(objs=sols_t)
    p_sols, p_sols_num, _ = nd_sort.show_PE()
    hvs = cal_ps_hv(pf=p_sols, pf_num=p_sols_num, ref=ref)

    print('Run Time(s): {:.4f}'.format(total_time))
    print('HV Ratio: {:.4f}'.format(hvs.mean()))

##########################################################################################
if __name__ == "__main__":
    main()