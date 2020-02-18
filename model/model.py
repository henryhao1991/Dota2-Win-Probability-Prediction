# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Other libraries
import numpy as np
from collections import OrderedDict
import math


class heuristic:
    """
    Heuristic model that just return the sigmoid of the difference in gold and experience combined,
    with scaling and bias.
    """
    
    def __init__(self, xp_scale_factor=1.0, total_scale_factor=2.0):
        """
        Keywords:
            xp_scale_factor: Float. The scaling factor between total gold and total experience.
            total_scale_factor: float. The scaling factor for the sum of total gold and experience before passing into sigmoid function.
        """
        self.xp_scale_factor = xp_scale_factor
        self.total_scale_factor = total_scale_factor
        self.gold_xp_scale = torch.cat((torch.ones(10),self.xp_scale_factor*torch.ones(10)))
        
    def fit(self, inputs):
        """
        Inputs:
            inputs: torch.Tensor. The input features for the heuristic model. Currently only support batch size of 1.
        Returns:
            Torch.Tensor. Returns the probability of radiant win.
        """
        gold_xp_combined = torch.sum(inputs,1)
        return self.sigmoid(gold_xp_combined/self.total_scale_factor)
    
    def sigmoid(self, inputs):
        return 1/(1+torch.exp(-inputs))


class LSTM_baseline(nn.Module):
    """
    LSTM model for win probability prediction.
    Bases: Torch.nn.Module
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim=1, batch_size=10, device=torch.device('cpu')):
        """
        Inputs:
            input_dim: int. Dimension of the input features.
            hidden_dim: int. Dimentsion of the hidden layer.
        Keywords:
            output_dim: int. Dimension of the output features. Should only be 1 since we are just predicting the probability.
            batch_size: int. Batch size used as the inputs.
            device: torch.device. The device used for the model.
        """
        super(LSTM_baseline, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        if device == torch.device('cuda'):
            if torch.cuda.is_available():
                print('GPU is used.')
                self.device = device
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.hidden_cell = self.init_hidden(batch_size)
        
    def init_hidden(self, batch_size):
        """
        The function to initialize the hidden layer.
        """
        return (torch.zeros(1,batch_size,self.hidden_dim).to(self.device),
               torch.zeros(1,batch_size,self.hidden_dim).to(self.device))
    
    def forward(self, inputs):
        """
        Inputs:
            inputs: torch.Tensor, size=(batch_size, input_dim, L). L is the length of the longest sequence in the batch.
        Returns:
            torch.Tensor, size=(batch_size, 1, L)
        """
        lstm_out, hidden_cell = self.lstm(inputs, self.hidden_cell)
        self.hidden_cell = tuple([hidden_.detach_() for hidden_ in hidden_cell])
        lstm_out = self.linear(lstm_out)
        return torch.tanh(lstm_out)
    

class LSTMWithH2vSubnet(nn.Module):
    """
    LSTM model plus a feed forward subnet for the team embeddings.
    Bases: Torch.nn.Module
    """
    def __init__(self, input_dim, hidden_dim, h2v_dim=20, h2v_layer_dim=[50,30,1], 
                 output_dim=1, batch_size=10, device=torch.device('cpu')):
        super(LSTMWithH2vSubnet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.h2v_dim = h2v_dim
        self.h2v_layer_dim = h2v_layer_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        if device == torch.device('cuda'):
            if torch.cuda.is_available():
                print('GPU is used.')
                self.device = device
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim+self.h2v_layer_dim[-1], self.output_dim)
        self.hidden_cell = self.init_hidden(batch_size)
        h2v_layers = OrderedDict()
        h2v_layers['Linear0'] = nn.Linear(h2v_dim, h2v_layer_dim[0])
        for i in range(len(h2v_layer_dim)-1):
            h2v_layers['Linear'+str(i+1)] = nn.Linear(h2v_layer_dim[i], h2v_layer_dim[i+1])
        self.h2v_linear = nn.Sequential(h2v_layers)
        
    def init_hidden(self, batch_size):
        """
        The function to initialize the hidden layer.
        """
        return (torch.zeros(1,batch_size,self.hidden_dim).to(self.device),
               torch.zeros(1,batch_size,self.hidden_dim).to(self.device))
    
    def forward(self, inputs, embeddings):
        lstm_out, hidden_cell = self.lstm(inputs, self.hidden_cell)
        self.hidden_cell = tuple([hidden_.detach_() for hidden_ in hidden_cell])
        h2v_out = self.h2v_linear(embeddings)
        concat_out = torch.cat((lstm_out, h2v_out.repeat(1,lstm_out.shape[1],1).view(-1,lstm_out.shape[1],1)), dim=2)
        concat_out = self.linear(concat_out)
        return torch.tanh(concat_out)
    
