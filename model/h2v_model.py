#PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim

#Other libraries
import numpy as np


class Hero2vecNetwork(nn.Module):
    '''
    Hero2vec network.
    Inspired by the following repo:
    https://github.com/ybw9000/hero2vec
    '''
    
    def __init__(self, embedding_dim, heropool_size=129):
        '''
        Inputs:
            embedding_dim: int. The number of embedding dimension.
        Keywords:
            heropool_size: int. The number of largest hero index. (Not the number of total heroes, since some index are not used currently in Dota2.)
        '''
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hero_embedding = nn.Embedding(heropool_size, self.embedding_dim)
        self.embedding_to_target = nn.Linear(self.embedding_dim, heropool_size)
        self.init_network()
        
    def init_network(self):
        '''
        Initiallize the network.
        Currently not used. Maybe will need it later.
        '''
        pass
    
    def forward(self, inputs):
        '''
        Inputs:
            inputs: torch.LongTensor, size=(4,)
        Returns:
            out: torch.LongTensor, size=(1,)
        '''
        embedding = self.hero_embedding(inputs).sum(dim=1)
        out = self.embedding_to_target(embedding)
        
        return out
