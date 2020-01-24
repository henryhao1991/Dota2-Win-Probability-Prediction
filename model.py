import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math


class heuristic:
    """
    Heuristic model that just return the sigmoid of the difference in gold and experience combined,
    with scaling and bias.
    """
    
    def __init__(self,xp_scale_factor=1.0,total_scale_factor=10.0):
        self.xp_scale_factor = xp_scale_factor
        self.total_scale_factor = total_scale_factor
        self.gold_xp_scale = torch.cat((torch.ones(10),self.xp_scale_factor*torch.ones(10)))
        
    def fit(self, inputs):
        gold_xp_combined = torch.sum(inputs,1)
        return self.sigmoid(gold_xp_combined/self.total_scale_factor)
    
    def sigmoid(self, inputs):
        return 1/(1+torch.exp(-inputs))


class LSTM_baseline(nn.Module):
    """
    LSTM model for win probability prediction.
    """
    
    def __init__(self,input_dim,hidden_dim,output_dim,batch_size=1,device=torch.device('cpu')):
        
        super(LSTM_baseline, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = device
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.hidden_cell = self.init_hidden()
        
    def init_hidden(self):
        return (torch.zeros(1,self.batch_size,self.hidden_dim).to(self.device),
               torch.zeros(1,self.batch_size,self.hidden_dim).to(self.device))
    
    def forward(self, inputs):
        lstm_out, hidden_cell = self.lstm(inputs, self.hidden_cell)
        self.hidden_cell = tuple([hidden_.detach_() for hidden_ in hidden_cell])
        lstm_out = self.linear(lstm_out)
        return lstm_out

