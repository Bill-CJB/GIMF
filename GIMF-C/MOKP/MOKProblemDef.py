import torch
import numpy as np

def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 3))
    # problems.shape: (batch, problem, 3)
    preference = torch.Tensor(batch_size, 2).uniform_(1e-6, 1)
    return problems, preference


