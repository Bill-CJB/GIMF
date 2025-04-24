from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)
    preference: torch.Tensor
    item_weight_img: torch.Tensor = None

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)

class KPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.problem_size = None
        self.pomo_size = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)
        self.preference = None

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        self.offsets = torch.tensor([[0, 0]])

    def load_problems(self, batch_size, problem_size, aug_factor=1, problems=None):
        self.batch_size = batch_size
        self.problem_size = problem_size
        self.pomo_size = problem_size

        if problems is not None:
            self.problems = problems
        else:
            from MOKP.MOKProblemDef import get_random_problems
            self.problems, self.preference = get_random_problems(batch_size, self.problem_size)

        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            raise NotImplementedError

        self.item_value = self.problems[:, :, 1:]
        # shape: (batch, problem, 2)
        self.item_weight = self.problems[:, :, 0]
        # shape: (batch, problem)

        self.item_weight_img = torch.ones((self.batch_size, self.channels, self.img_size, self.img_size))
        xy_img = self.item_value * self.img_size
        xy_img = xy_img.int()
        block_indices = xy_img // self.patch_size
        self.block_indices = block_indices[:, :, 0] * self.patches + block_indices[:, :, 1]

        xy_img = xy_img[:, None, :, None, :] + self.offsets[None, None, None, :, :].expand(self.batch_size, self.channels, 1,
                                                                                           self.offsets.shape[0],
                                                                                           self.offsets.shape[
                                                                                               1]).contiguous()
        xy_img_idx = xy_img.view(-1, 2)
        weight = self.item_weight[:, None, :, None, None].expand(
            self.batch_size, self.channels, self.pomo_size, self.offsets.shape[0], 1).contiguous().view(-1)
        CHANEl_IDX = torch.arange(self.channels)[None, :, None, None].expand(self.batch_size, self.channels, self.problem_size,
                                                                 self.offsets.shape[0]).contiguous().view(-1)
        BATCH_IDX = torch.arange(self.batch_size)[:, None, None, None].expand(self.batch_size, self.channels, self.problem_size,
                                                                              self.offsets.shape[0]).contiguous().view(
            -1)
        self.item_weight_img[BATCH_IDX, CHANEl_IDX, xy_img_idx[:, 0], xy_img_idx[:, 1]] = weight

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        
        # MOKP
        ###################################
        self.items_and_a_dummy = torch.Tensor(np.zeros((self.batch_size, self.problem_size+1, 3)))
        self.items_and_a_dummy[:, :self.problem_size, :] = self.problems
        self.item_data = self.items_and_a_dummy[:, :self.problem_size, :]

        if self.problem_size >= 50 and self.problem_size < 100:
            capacity = 12.5
        elif self.problem_size >= 100 and self.problem_size <= 200:
            capacity = 25
        else:
            raise NotImplementedError
        self.capacity = torch.Tensor(np.ones((self.batch_size, self.pomo_size))) * capacity
        
        self.accumulated_value_obj1 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))
        self.accumulated_value_obj2 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))
        
        self.ninf_mask_w_dummy = torch.zeros(self.batch_size, self.pomo_size, self.problem_size+1)
        self.ninf_mask = self.ninf_mask_w_dummy[:, :, :self.problem_size]
        
        self.fit_ninf_mask = None
        self.finished = torch.BoolTensor(np.zeros((self.batch_size, self.pomo_size)))
       

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
       
        # MOKP
        ###################################
        self.items_and_a_dummy = torch.Tensor(np.zeros((self.batch_size, self.problem_size+1, 3)))
        self.items_and_a_dummy[:, :self.problem_size, :] = self.problems
        self.item_data = self.items_and_a_dummy[:, :self.problem_size, :]

        if self.problem_size >= 50 and self.problem_size < 100:
            capacity = 12.5
        elif self.problem_size >= 100 and self.problem_size <= 200:
            capacity = 25
        else:
            raise NotImplementedError
        self.capacity = torch.Tensor(np.ones((self.batch_size, self.pomo_size))) * capacity
        
        self.accumulated_value_obj1 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))
        self.accumulated_value_obj2 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))
       
        self.ninf_mask_w_dummy = torch.zeros(self.batch_size, self.pomo_size, self.problem_size+1)
        self.ninf_mask = self.ninf_mask_w_dummy[:, :, :self.problem_size]
        
        self.fit_ninf_mask = None
        self.finished = torch.BoolTensor(np.zeros((self.batch_size, self.pomo_size)))
       
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        self.step_state.capacity = self.capacity
        self.step_state.finished = self.finished

        reward = None
        done = False
        return Reset_State(self.problems, self.preference, self.item_weight_img), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
       
        # Status
        ####################################
        items_mat = self.items_and_a_dummy[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size+1, 3)
        gathering_index = selected[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, 3)
        selected_item = items_mat.gather(dim=2, index=gathering_index).squeeze(dim=2)
       
        self.accumulated_value_obj1 += selected_item[:, :, 1]
        self.accumulated_value_obj2 += selected_item[:, :, 2]
        self.capacity -= selected_item[:, :, 0]

        batch_idx_mat = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        group_idx_mat = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.ninf_mask_w_dummy[batch_idx_mat, group_idx_mat, selected] = -np.inf

        unfit_bool = (self.capacity[:, :, None] - self.item_data[:, None, :, 0]) < 0
        self.fit_ninf_mask = self.ninf_mask.clone()
        self.fit_ninf_mask[unfit_bool] = -np.inf

        self.finished = (self.fit_ninf_mask == -np.inf).all(dim=2)
        done = self.finished.all()
        self.fit_ninf_mask[self.finished[:, :, None].expand(self.batch_size, self.pomo_size, self.problem_size)] = 0
       
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.fit_ninf_mask
        self.step_state.capacity = self.capacity
        self.step_state.finished = self.finished
        
        reward = None
        if done:
            reward = torch.stack([self.accumulated_value_obj1,self.accumulated_value_obj2],axis = 2)
        else:
            reward = None

        return self.step_state, reward, done

   
