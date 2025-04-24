import torch

import os
from logging import getLogger

from MOKPEnv import KPEnv as Env
from MOKPModel import KPModel as Model

from MOKProblemDef import get_random_problems

from einops import rearrange
import math
from utils.utils import *


class KPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params,
                 logger=None,
                 result_folder=None,
                 checkpoint_dict=None,
                 ):

        
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        if logger:
            self.logger = logger
            self.result_folder = result_folder
        else:
            self.logger = getLogger(name='trainer')
            self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env()
        self.model = Model(**self.model_params)
        
        if checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict['model_state_dict'])
        else:
            model_load = tester_params['model_load']
            checkpoint_fullname = '{path}/checkpoint_mokp-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self, shared_problem, pref, episode=0):
        self.time_estimator.reset()
    
        score_AM = {}
      
        # 2 objs
        for i in range(2):
            score_AM[i] = AverageMeter()
          
        test_num_episode = self.tester_params['test_episodes']
        episode = episode
        
        while episode < test_num_episode:
            
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score = self._test_one_batch(shared_problem, pref, batch_size, episode)
            
            # 2 objs
            for i in range(2):
                score_AM[i].update(score[i], batch_size)
               
            episode += batch_size
    
            ############################
            # Logs
            ############################

            self.logger.info("AUG_OBJ_1 SCORE: {:.4f}, AUG_OBJ_2 SCORE: {:.4f} ".format(score_AM[0].avg.mean(), score_AM[1].avg.mean()))
            break
        return [score_AM[0].avg.cpu(), score_AM[1].avg.cpu()]

    def _test_one_batch(self, shared_probelm, pref, batch_size, episode):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        self.env.problem_size = self.env_params['problem_size']
        self.env.pomo_size = self.env_params['pomo_size']

        img_size = math.ceil(self.env.problem_size ** (1 / 2) * self.model_params['pixel_density'] / self.model.model_params['patch_size']) * self.model.model_params['patch_size']
        self.env.channels = self.model_params['in_channels']
        self.env.img_size = img_size
        self.env.patch_size = self.model_params['patch_size']
        self.env.patches = self.env.img_size // self.env.patch_size
        self.model.encoder.embedding_patch.patches = self.env.patches
        self.model.decoder.patches = self.env.patches

        problems = shared_probelm[episode: episode + batch_size]
        self.env.preference = pref[episode: episode + batch_size]
        self.env.load_problems(batch_size, self.env_params['problem_size'], aug_factor, problems)

        
        self.model.eval()
        with torch.no_grad():
            reset_state, _, _ = self.env.reset()
            pref = reset_state.preference

            self.model.pre_forward(reset_state, pref)
            
        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, _ = self.model(state)
            
            action_w_finished = selected.clone()
            action_w_finished[state.finished] = self.env_params['problem_size']  # this is dummy item with 0 size 0 value
            
            state, reward, done = self.env.step(action_w_finished)

        a = pref[:, 0]
        b = pref[:, 1]
        x = 1 / (1 + b / a)
        y = 1 - x
        preference = torch.cat((x[:, None], y[:, None]), dim=-1)
        new_pref = preference[:, None, :].expand_as(reward)

        if self.tester_params['dec_method'] == 'WS':
            tch_reward = (new_pref * reward).sum(dim=2)
        else:
            return NotImplementedError

        
        group_reward = tch_reward
        _ , max_idx = group_reward.max(dim=1)
        max_idx = max_idx.reshape(max_idx.shape[0],1)
        max_reward_obj1 = reward[:,:,0].gather(1, max_idx)
        max_reward_obj2 = reward[:,:,1].gather(1, max_idx)
    
        score = []
        
        score.append(max_reward_obj1.float())
        score.append(max_reward_obj2.float())

        return score

     

