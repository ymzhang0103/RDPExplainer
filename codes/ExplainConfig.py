import argparse
import numpy as np
import os
import random
import torch

from GNNConfig import model_args

def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default='datasets', help="Dataset root dir string")  
    parser.add_argument("--dataset", type=str, default='BA_shapes', help="Dataset string")# BA_shapes,   BA_community
    parser.add_argument('--seed',type=int, default=1234, help='seed')
    parser.add_argument('--dtype', type=str, default='float32')  
    parser.add_argument("--elr", type=float, default=0.03,help='initial learning rate.')
    parser.add_argument('--coff_size', type=float, default=0.01, help='coefficient for size constriant') 
    parser.add_argument('--coff_diff', type=float, default=10.0, help='coefficient for value different') 
    parser.add_argument('--eepochs', type=int, default=100, help='Number of epochs to train explainer.')
    parser.add_argument('--budget', type=float, default=-1.0, help='coefficient for size constriant')
    #parser.add_argument('--coff_ent', type=float, default=1.0, help='coefficient for entropy loss') #0.01
    #parser.add_argument('--weight_decay',type=float, default=0.0, help='Weight for L2 loss on embedding matrix.')
    #parser.add_argument('--coff_t0', type=float, default=5.0, help='initial temperature')
    #parser.add_argument('--coff_te', type=float, default=2.0, help='final temperature')
    parser.add_argument('--sample_bias',type=float, default=0.0, help='bias for sampling from 0-1')
    parser.add_argument('--maskmodel',type=str, default='GAT', help='mask generator')
    parser.add_argument('--loss_flag',type=str, default='GAT', help='loss function')
    #parser.add_argument('--tmp',type=float, default=0.05, help='parameter tmp')
    parser.add_argument('--tau0',type=float, default=1.0, help='parameter tau0')
    parser.add_argument('--tauT',type=float, default=0.05, help='parameter tauT')
    parser.add_argument('--tau1',type=float, default=0.00001, help='parameter tau1')

    args, _ = parser.parse_known_args()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.latent_dim = model_args.latent_dim
    return args

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

args = get_params()
set_seed(args.seed)

dtype = torch.float32
if args.dtype=='float64':
    dtype = torch.float64
