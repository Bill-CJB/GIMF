##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 4


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

from MOTSPTester_3obj import TSPTester as Tester
from MOTSProblemDef_3obj import get_random_problems
# from generate_test_dataset import load_dataset

##########################################################################################
import time

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    'in_channels': 3,
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
        'path': './result/train__tsp',  # directory path of pre-trained model and log files saved.
        'epoch': 200,
    },
    'test_episodes': 200,
    'test_batch_size': 200,
    'augmentation_enable': True,
    'aug_factor': 1,
    'aug_batch_size': 200,
    'n_sols': 105  #1035, 10011
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp',
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


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)
            
def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)

##########################################################################################
timer_start = time.time()
logger_start = time.time()

def main(n_sols = tester_params['n_sols']):
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

    test_path = f"./data/testdata_tsp_3o_size{problem_size}.pt"
    shared_problem = torch.load(test_path).to(device=CUDA_DEVICE_NUM)

    if n_sols == 105:
        uniform_weights = torch.Tensor(das_dennis(13,3))  # 105
    elif n_sols == 1035:
        uniform_weights = torch.Tensor(das_dennis(44,3))   # 1035
    elif n_sols == 10011:
        uniform_weights = torch.Tensor(das_dennis(140,3))   # 10011


    batch_size = shared_problem.shape[0]
    sols = np.zeros([batch_size, n_sols, 3])
    mini_batch_size = tester_params['test_batch_size']
    b_cnt = tester_params['test_episodes'] / mini_batch_size
    b_cnt = int(b_cnt)

    for bi in range(0, b_cnt):
        b_start = bi * mini_batch_size
        b_end = b_start + mini_batch_size
        for i in range(n_sols):
            pref = uniform_weights[i]
            pref = pref[None, :].expand(shared_problem.size(0), 3)
            aug_score = tester.run(shared_problem, pref, episode=b_start)

            # sols[i] = np.array(aug_score)
            sols[b_start:b_end, i, 0] = np.array(aug_score[0].flatten())
            sols[b_start:b_end, i, 1] = np.array(aug_score[1].flatten())
            sols[b_start:b_end, i, 2] = np.array(aug_score[2].flatten())

    timer_end = time.time()
    
    total_time = timer_end - timer_start

    
    if problem_size == 20:
        ref = np.array([20,20,20])    #20
    elif problem_size == 50:
        ref = np.array([35,35,35])   #50
    elif problem_size == 100:
        ref = np.array([65,65,65])   #100
    else:
        print('Have yet define a reference point for this problem size!')


    nd_sort = Pareto_sols(p_size=problem_size, pop_size=sols.shape[0], obj_num=sols.shape[2])
    sols_t = torch.Tensor(sols)
    nd_sort.update_PE(objs=sols_t)
    p_sols, p_sols_num, _ = nd_sort.show_PE()
    hvs = cal_ps_hv(pf=p_sols, pf_num=p_sols_num, ref=ref)
    
    print('Run Time(s): {:.4f}'.format(total_time))
    print('HV Ratio: {:.4f}'.format(hvs.mean()))
    # print(hv_ratio)

##########################################################################################

if __name__ == "__main__":
    main()
