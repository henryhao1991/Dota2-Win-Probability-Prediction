from dataloader import *
from model import heuristic
import numpy as np
import torch
from matplotlib import pyplot as plt


class heuristic_run:
    """
    The class to run the heuristic model. No training needed.
    Parameters can be tuned in model.heuristic
    """
    def __init__(self):
        self.dataloader = single_dataloader()
        self.num_data = len(self.dataloader)
        self.model = heuristic()
        
    def get_inputs_and_results(self, features, labels):
        
        rad_dire = torch.cat((torch.ones(5),-1.0*torch.ones(5),torch.ones(5),-1.0*torch.ones(5)))
        inputs = features.squeeze()[:,:20].to(torch.float) * rad_dire.to(torch.float)
        result = labels.squeeze()[0]
        
        return (inputs, result)

    def get_accuracy(self, threshold=0.25):
        
        acc = 0.0
        for (features, labels) in self.dataloader:
            
            inputs, result = self.get_inputs_and_results(features, labels)
            output = self.model.fit(inputs)[-1]
            if (output-0.5)*result > threshold:
                acc += 1/self.num_data
                
        return acc
    
    def get_single_graph(self, ind):
        
        features, labels = self.dataloader.dataset[ind]
        inputs, _ = self.get_inputs_and_results(torch.tensor(features), labels)
        outputs = self.model.fit(inputs)
        
        return outputs
