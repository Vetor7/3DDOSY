import numpy as np
import torch
from config import parse_args
from train import train
def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
if __name__ == '__main__':
    set_random_seeds(42)
    args = parse_args()
    train(args, need_generate=False)