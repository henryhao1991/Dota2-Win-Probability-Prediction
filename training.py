from dataloader import *
from model import *
import numpy as np
from math import ceil, floor
import torch
from matplotlib import pyplot as plt

class TrainingAndEvaluation:
    """
    The class to train and evaluate the result based on the selected model.
    """
    def __init__(self, *args, model='LSTM_baseline', **kwargs):
        self.model_name = {'heuristic':HeuristicTrain, 'LSTM_baseline':LSTMBaselineTrain}
        assert model in self.model_name.keys(), "Invalid model name."
        self.model = self.model_name[model](*args, **kwargs)
        self.num_test_data = len(self.model.dataloader["test"])
        
    def train(self):
        return self.model.train()
        
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
    def __init__(self, xp_scale_factor=1.0, total_scale_factor=10.0, **kwargs):
        self.dataloader = split_dataloader(PreprocessedParsedReplayDataset, **kwargs)
        self.model = heuristic(xp_scale_factor, total_scale_factor)
        
    def preprocess(self, features, labels):
        
        rad_dire = torch.cat((torch.ones(5),-1.0*torch.ones(5),torch.ones(5),-1.0*torch.ones(5)))
        inputs = features.squeeze()[:,:20].to(torch.float) * rad_dire.to(torch.float)
        
        return inputs
    
    def train(self):
        print("This is a heuristic model. No training needed.")
        return None
    
    def get_lengths_and_labels(self, batch_data):
        
        end_labels = batch_data["labels"][:,-1].numpy()
        results = np.where(end_labels>0, 1.0, -1.0)
        return {"lengths":batch_data["lengths"], "results":results}

    def predict(self, batch_data, threshold=0.25):
        
        inputs = self.preprocess(batch_data["features"], batch_data["labels"])
        predictions = self.model.fit(inputs)
 
        return predictions


class LSTMBaselineTrain:
    """
    The class to train the LSTM baseline model and make predictions.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_epochs=100, lr=0.01, loss_function=nn.MSELoss(), 
                 batch_size=10, device=torch.device('cpu'), **kwargs):
        self.dataloader = split_dataloader(PreprocessedParsedReplayDataset, batch_size=batch_size, **kwargs)
        self.model = LSTM_baseline(input_dim, hidden_dim, output_dim=output_dim, batch_size=batch_size, device=device)
        self.num_epochs = num_epochs
        self.loss_function = loss_function
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        
    def train(self, epoch_print=10):
        ave_train_loss = []
        ave_val_loss = []
        for epoch in range(self.num_epochs):
    
            total_loss = 0
            total_val_loss = 0

            train_count = 0
            for batch in self.dataloader['train']:
                sequences, labels = batch["features"].to(self.model.device), batch["labels"].to(self.model.device)

                train_count += 1
                self.model.hidden_cell = self.model.init_hidden()
                self.model.zero_grad()

                outputs = self.model(sequences).squeeze()
                loss = self.loss_function(outputs, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss


            val_count = 0
            with torch.no_grad():
                for batch in self.dataloader['val']:
                    sequences, labels = batch["features"].to(self.model.device), batch["labels"].to(self.model.device)
                    val_count += 1

                    outputs = self.model(sequences)
                    loss = self.loss_function(outputs.squeeze(), labels)

                    total_val_loss += loss

            ave_train_loss.append(total_loss/train_count)
            ave_val_loss.append(total_val_loss/val_count)
            
            if (epoch+1)%epoch_print == 0:
                print("Epoch %d finished. Train loss: %f. Validation loss: %f" % 
                     (epoch+1, ave_train_loss[-1], ave_val_loss[-1]))
        return {"train_loss":ave_train_loss, "val_loss":ave_val_loss}
    
    def get_lengths_and_labels(self, batch_data):
        
        end_labels = batch_data["labels"][:,-1].numpy()
        results = np.where(end_labels>0, 1.0, -1.0)
        return {"lengths":batch_data["lengths"], "results":results}

    def predict(self, batch_data, threshold=0.25):
        
        with torch.no_grad():
            dim = batch_data["features"].shape
            if len(dim) == 3:
                self.model.hidden_cell = self.model.init_hidden()
                predictions = self.model(batch_data["features"])
            else:
                self.model.batch_size = 1
                self.model.hidden_cell = self.model.init_hidden()
                predictions = self.model(batch_data["features"].view(1,dim[0],dim[1]))
 
        return predictions / 2.0 + 0.5
    
