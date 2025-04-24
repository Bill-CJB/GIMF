import torch

import os
from logging import getLogger

from MOTSPEnv_3obj import TSPEnv as Env
from MOTSPModel_3obj import TSPModel as Model

from MOTSProblemDef_3obj import augment_xy_data_by_n_fold_3obj, augment_preference
import math
from einops import rearrange

from utils.utils import *


class TSPTester:
    def __init__(self,
                 model_params,
                 tester_params):

        # save arguments
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
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
        
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint_motsp-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self, shared_problem, pref, episode=0):
        self.time_estimator.reset()
        
        aug_score_AM = {}
        
        # 3 objs
        for i in range(3):
            aug_score_AM[i] = AverageMeter()


        batch_size = self.tester_params['test_batch_size']

        aug_score = self._test_one_batch(shared_problem, pref, batch_size, episode)

        # 3 objs
        for i in range(3):
            aug_score_AM[i].update(aug_score[i], batch_size)

        ############################
        # Logs
        ############################
        self.logger.info("AUG_OBJ_1 SCORE: {:.4f}, AUG_OBJ_2 SCORE: {:.4f}, AUG_OBJ_3 SCORE: {:.4f} ".format(
            aug_score_AM[0].avg.mean(), aug_score_AM[1].avg.mean(), aug_score_AM[2].avg.mean()))
            
        return [aug_score_AM[0].avg.cpu(), aug_score_AM[1].avg.cpu(), aug_score_AM[2].avg.cpu()]
              
    def _test_one_batch(self, shared_probelm, pref, batch_size, episode):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
        _, problem_size, _ = shared_probelm.size()


        img_size = math.ceil(problem_size ** (1 / 2) * self.model_params['pixel_density'] / self.model.model_params['patch_size']) * self.model.model_params['patch_size']
        self.env.channels = self.model_params['in_channels']
        self.env.img_size = img_size
        self.env.patch_size = self.model_params['patch_size']
        self.env.patches = self.env.img_size // self.env.patch_size
        self.model.encoder.embedding_patch.patches = self.env.patches
        self.model.decoder.patches = self.env.patches

        problems = shared_probelm[episode: episode + batch_size]
        self.env.preference = pref[episode: episode + batch_size]
        self.env.load_problems(batch_size, problem_size, aug_factor, problems)
      
        self.model.eval()
        with torch.no_grad():
            reset_state, _, _ = self.env.reset()
            instances = reset_state.instances
            pref = reset_state.preference
            img = reset_state.xy_img
            self.model.pre_forward(instances, pref, img)

        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # reward was negative, here we set it to positive to calculate TCH
        reward = - reward


        a = pref[:, 0]
        b = pref[:, 1]
        c = pref[:, 2]
        x = 1 / (1 + b / a + c / a)
        y = (b / a) * x
        w = (c / a) * x
        preference = torch.cat((x[:, None], y[:, None], w[:, None]), dim=-1)
        pref = preference[:, None, :].expand_as(reward)
        tch_reward = (pref * reward).sum(dim=2)


        reward = - reward
        tch_reward = -tch_reward

        tch_reward = tch_reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        
        tch_reward_aug = rearrange(tch_reward, 'c b h -> b (c h)') 
        _ , max_idx_aug = tch_reward_aug.max(dim=1)
        max_idx_aug = max_idx_aug.reshape(max_idx_aug.shape[0],1)
        max_reward_obj1 = rearrange(reward[:,:,0].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_aug)
        max_reward_obj2 = rearrange(reward[:,:,1].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_aug)
        max_reward_obj3 = rearrange(reward[:,:,2].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_aug)
     
        aug_score = []

        aug_score.append(-max_reward_obj1.float())
        aug_score.append(-max_reward_obj2.float())
        aug_score.append(-max_reward_obj3.float())
        
        return aug_score

       