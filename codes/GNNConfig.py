import os
import torch
from tap import Tap
from typing import List


class DataParser(Tap):
    #dataset_name: str = 'bbbp'
    #dataset_name: str = 'BA_shapes'
    #dataset_name: str = 'BA_community'
    #dataset_name: str = 'Graph-Twitter' 
    dataset_name: str = 'Graph-SST5' 
    dataset_dir: str = 'datasets'
    random_split: bool = True
    data_split_ratio: List = [0.8, 0.1, 0.1]   # the ratio of training, validation and testing set for random split
    seed: int = 1


class GATParser(Tap):           # hyper-parameter for gat model
    gat_dropout: float = 0.6    # dropout in gat layer
    gat_heads: int = 10         # multi-head
    gat_hidden: int = 10        # the hidden units for each head
    gat_concate: bool = True    # the concatenation of the multi-head feature
    num_gat_layer: int = 3      # the gat layers


class ModelParser(GATParser):
    #device_id: int = 1
    model_name: str = 'gcn'
    checkpoint: str = './checkpoint'
    concate: bool = False                     # whether to concate the gnn features before mlp
    latent_dim: List[int] = [20, 20, 20]          # the hidden units for each gnn layer[20, 20, 20]      [128, 128, 128]   
    readout: 'str' = 'max'                    # the graph pooling method
    mlp_hidden: List[int] = []                # the hidden units for mlp classifier
    gnn_dropout: float = 0.0                  # the dropout after gnn layers
    dropout: float = 0.6                      # the dropout after mlp layers
    adj_normlize: bool = True                 # the edge_weight normalization for gcn conv
    #emb_normlize: bool = False                # the l2 normalization after gnn layer
    emb_normlize: bool = True
    model_path: str = ""                      # default path to save the model

    def process_args(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if not self.model_path:
            self.model_path = os.path.join(self.checkpoint,
                                           DataParser().parse_args(known_only=True).dataset_name,
                                           f"{self.model_name}_best.pth")


class TrainParser(Tap):
    learning_rate: float = 0.005
    batch_size: int = 64
    #batch_size: int = 1
    weight_decay: float = 0.0
    max_epochs: int = 800    #800
    save_epoch: int = 10
    early_stopping: int = 800 #800



data_args = DataParser().parse_args(known_only=True)
model_args = ModelParser().parse_args(known_only=True)
train_args = TrainParser().parse_args(known_only=True)


import random
import numpy as np
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
