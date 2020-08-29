import torch
import numpy as np
import torch.nn as nn
from clustering import Clustering
from autoencoder import AutoEncoder


class DeepClusteringNetwork(nn.Module):
    
    def __init__(self, args):
        super(self, DeepClusteringNetwork).__init__()
        self.args = args
        self.autoencoder = AutoEncoder(args)
        self.clustering = Clustering(args)
        
        self.criterion  = nn.MSELoss()
        self.optimizer = torch.optim.Adam(lr=args.lr,
                                          weight_decay=args.weight_decay)
    
    def pretrain(self, X):
        pass