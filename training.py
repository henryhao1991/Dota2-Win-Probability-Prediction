from dataloader import *
from model import heuristic
import numpy as np
from math import ceil, floor
import torch
from matplotlib import pyplot as plt

class TrainingAndEvaluation:
    """
    The class to train and evaluate the result based on the selected model.
    """
    def __init__(self, model='LSTM_baseline'):
        self.model_name = {'heuristic':HeuristicTrain, 'LSTM_baseline':LSTMBaselineTrain}
        assert model in self.model_name.keys(), "Invalid model name."
        self.model = self.model_name[model]()
        self.num_test_data = len(self.model.dataloader["test"])
        
    def get_accuracy(self, threshold=0.1, percentage=0.05):
        
        number_of_slices = ceil(1.0/percentage)
        acc_array = np.zeros(int(number_of_slices))
        
        for data in self.model.dataloader["test"]:
            
            lengths_and_results = self.model.get_lengths_and_labels(data)
            batch_lengths, batch_results = lengths_and_results["lengths"], lengths_and_results["results"]

            batch_predictions = self.model.predict(data)
            if len(batch_predictions.shape) == 1:
                batch_predictions = batch_predictions.view(1,batch_predictions.shape[0])
            
            for i in range(len(batch_lengths)):

                percent_length = batch_lengths[i]/number_of_slices
                
                percentage = 0
                for t in range(int(batch_lengths[i])):
                    
                    if t+1 > percentage*percent_length:
                        if (batch_predictions[i][t]-0.5) * batch_results[i] > threshold:
                            acc_array[percentage] += 1.0/self.num_test_data
                        percentage += 1
                            
        return acc_array
    
    def get_single_game_prediction(self, ind):
        data = self.model.dataloader["test"].dataset[ind]
        prediction = self.model.predict(data)
        
        return prediction


class HeuristicTrain:
    """
    The class to run the heuristic model. No training needed.
    Parameters can be tuned in model.heuristic
    """
    def __init__(self):
        self.dataloader = split_dataloader(PreprocessedParsedReplayDataset, batch_size=1, p_val=0.1, p_test = 0.2, 
                                           shuffle=False, collate_fn=PadSequence)
        self.model = heuristic()
        
    def preprocess(self, features, labels):
        
        rad_dire = torch.cat((torch.ones(5),-1.0*torch.ones(5),torch.ones(5),-1.0*torch.ones(5)))
        inputs = features.squeeze()[:,:20].to(torch.float) * rad_dire.to(torch.float)
        
        return inputs
    
    def get_lengths_and_labels(self, batch_data):
        
        end_labels = batch_data["labels"][:,-1].numpy()
        results = np.where(end_labels>0, 1.0, -1.0)
        return {"lengths":batch_data["lengths"], "results":results}

    def predict(self, batch_data, threshold=0.25):
        
        inputs = self.preprocess(batch_data["features"], batch_data["labels"])
        predictions = self.model.fit(inputs)
 
        return predictions


class LSTMBaselineTrain:
    pass