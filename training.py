# PyTorch imports
import torch

# Custom Python files
from dataloader import *
from model import *

# Other libraries
import numpy as np
from math import ceil, floor

class TrainingAndEvaluation:
    """
    The class to train and evaluate the result based on the selected model.
    """
    def __init__(self, *args, model='LSTM_with_h2v', **kwargs):
        """
        Inputs:
            *args: the arguments passing to the selected model.
        Keywords:
            model: string. The name of the selected model.
            **kwargs: the keywords passing to the selected model.
        """
        self.model_name = {'heuristic':HeuristicTrain, 'LSTM_baseline':LSTMBaselineTrain, 'LSTM_with_h2v':LSTMWithH2vTrain}
        assert model in self.model_name.keys(), "Invalid model name."
        
        self.model = self.model_name[model](*args, **kwargs)
        
        #Get the total number of test data
        self.num_test_data = len(self.model.dataloader["test"])
        
    def train(self):
        return self.model.train()
        
    def get_accuracy(self, threshold=0.1, percentage=0.05):
        """
        Returns:
            acc_array: numpy.ndarray. The array of the accuracy at different percentage of the game.
        Keywords:
            threshold: float. The threshold that we want to predict the winner. E.g., if set to be 0.1, >0.6 will be predicted as radiant win and <0.4 will be predicted as dire win.
            percentage: float. The percentage interval to check for accuracy.
        """
        #Find the number of time slices that we want to check for accuracy, and initialize the output accuracy array to 0.
        number_of_slices = ceil(1.0/percentage)
        acc_array = np.zeros(int(number_of_slices))
        
        #Loop through the test set.
        for data in self.model.dataloader["test"]:
            
            #Get the lengths and final results of games in the batch.
            lengths_and_results = self.model.get_lengths_and_labels(data)
            batch_lengths, batch_results = lengths_and_results["lengths"], lengths_and_results["results"]
            
            #Get predictions made by the model.
            batch_predictions = self.model.predict(data)
            
            #Add a new dimension of 1 at the beginning if batch_size is 1.
            if len(batch_predictions.shape) == 1:
                batch_predictions = batch_predictions.view(1,batch_predictions.shape[0])
            
            for i in range(len(batch_lengths)):
                
                #Find the time stamps that we want to check accuracy.
                percent_length = batch_lengths[i]/number_of_slices
                
                percentage = 0
                
                #Loop through the games in the batch.
                for t in range(int(batch_lengths[i])):
                    
                    #Calculate the accuracy if we reach a time stamp.
                    if t+1 > percentage*percent_length:
                        if (batch_predictions[i][t]-0.5) * batch_results[i] > threshold:
                            acc_array[percentage] += 1.0/self.num_test_data
                        percentage += 1
                            
        return acc_array
    
    def get_single_game_prediction(self, ind):
        """
        Inputs:
            ind: int. The index of the game that we want to get the graph.
        Returns:
            prediction: torch.Tensor. the prediction of probability that radiant will win with 30 sec interval.
        """
        data = self.model.dataloader["test"].dataset[ind]
        prediction = self.model.predict(data)
        
        return prediction
    
    def get_prediction_from_file(self, nparray_from_file):
        
        data = {"features":torch.tensor(nparray_from_file, dtype=torch.float)}
        prediction = self.model.predict(data)
        print(type(data['features']))
        
        return prediction


class HeuristicTrain:
    """
    The class to run the heuristic model. No training needed.
    Parameters can be tuned in model.heuristic
    """
    def __init__(self, xp_scale_factor=1.0, total_scale_factor=2.0, **kwargs):
        """
        Keywords:
            xp_scale_factor: float. The keyword passing to model.heuristic class.
            total_scale_factor: float. The keyword passing to model.heuristic class.
            **kwargs: The keywords passing to dataloader.split_dataloader function.
        """
        self.dataloader = split_dataloader(PreprocessedParsedReplayDataset, **kwargs)
        self.model = heuristic(xp_scale_factor, total_scale_factor)
        
    def preprocess(self, features):
        """
        Preprocess the data to use for the heuristic model.
        
        Inputs:
            features: torch.Tensor, size=(batch_size, *, input_dim). * is the length of the game. Input features that needs to be preprocessed. Only batch_size of 1 is supported currently.
        Returns:
            processed_inputs: torch.Tensor, size=(*, 20). * is the length of the game. Preprocessed features for the heuristic model. Only the individual gold and experience are used. positive for radiant, and negative for dire.
        """
        rad_dire = torch.cat((torch.ones(5),-1.0*torch.ones(5),torch.ones(5),-1.0*torch.ones(5)))
        processed_inputs = features.squeeze()[:,:20].to(torch.float) * rad_dire.to(torch.float)
        
        return processed_inputs
    
    def train(self):
        """
        The heuristic model does not train!
        """
        print("This is a heuristic model. No training needed.")
        return None
    
    def get_lengths_and_labels(self, batch_data):
        """
        Get the lengths and the final results of games.
        
        Inputs:
            batch_data: torch.util.data.Dataset. Should be dataloader.PreprocessedParsedReplayDataset class.
        Returns:
            dictionary:
                        "lengths":torch.Tensor, size(batch_size). Lengths of the games in batch (Should only be 1 now though).
                        "results":numpy.ndarray, size(batch_size). Results of the games in the batch (Should only be 1 now though). -1 if dire winned and +1 if radiant winned.
        """
        end_labels = batch_data["labels"][:,-1].numpy()
        results = np.where(end_labels>0, 1.0, -1.0)
        return {"lengths":batch_data["lengths"], "results":results}

    def predict(self, batch_data, threshold=0.25):
        """
        Make predictions with a 30 second interval.
        
        Inputs:
            batch_data: torch.util.data.Dataset. Should be dataloader.PreprocessedParsedReplayDataset class.
        Returns:
            predictions: torch.Tensor, size=(batch_size, L). Predictions of the probability that radiant will win. batch_size should only be 1. L is the length of the game.
        """
        inputs = self.preprocess(batch_data["features"])
        predictions = self.model.fit(inputs)
 
        return predictions


class LSTMBaselineTrain:
    """
    The class to train the LSTM baseline model and make predictions.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_epochs=100, lr=0.01, loss_function=nn.MSELoss(), 
                 batch_size=10, device=torch.device('cpu'), **kwargs):
        """
        Inputs:
            input_dim, hidden_dim: arguments passing to model.LSTM_baseline class.
        Keywords:
            output_dim, batch_size, device: keywords passing to model.LSTM_baseline class.
            num_epoches: int. Number of epochs used for training.
            lr: float. Learning rate used for training.
            loss_function: torch.nn loss function. Loss function used for the training.
            **kwargs: keywords passing to dataloader.split_dataloader function.
        """
        self.dataloader = split_dataloader(PreprocessedParsedReplayDataset, batch_size=batch_size, **kwargs)
        self.model = LSTM_baseline(input_dim, hidden_dim, output_dim=output_dim, batch_size=batch_size, device=device)
        if self.model.device == torch.device('cuda'):
            self.model.cuda()
        self.num_epochs = num_epochs
        self.loss_function = loss_function
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        
    def train(self, epoch_print=10):
        """
        The function to train the model.
        
        Returns:
            dictionary:
                    "train_loss":list. Average training loss at each epoch.
                    "val_loss":list. Average validation loss at each epoch.
        Keywords:
            epoch_print: int. Number of epochs between printing out the train and validation loss.
        """
        #List of verage training and validation loss
        ave_train_loss = []
        ave_val_loss = []
        
        #For each epoch
        for epoch in range(self.num_epochs):
            
            #Training and validation loss for this single epoch.
            total_loss = 0
            total_val_loss = 0
            
            #Count the number of training data
            train_count = 0
            for batch in self.dataloader['train']:
                sequences, labels = batch["features"].to(self.model.device), batch["labels"].to(self.model.device)

                train_count += 1
                
                #Initialize the hidden layer and zero the stored gradient.
                self.model.hidden_cell = self.model.init_hidden(self.model.batch_size)
                self.model.zero_grad()

                outputs = self.model(sequences).squeeze()
                loss = self.loss_function(outputs, labels)
                
                #Backprop and optimize.
                loss.backward()
                self.optimizer.step()

                total_loss += loss

            #Count the number of validation data
            val_count = 0
            
            #Do not store gradiant, since we are not training.
            with torch.no_grad():
                for batch in self.dataloader['val']:
                    
                    sequences, labels = batch["features"].to(self.model.device), batch["labels"].to(self.model.device)
                    val_count += 1
                    
                    self.model.hidden_cell = self.model.init_hidden(self.model.batch_size)

                    outputs = self.model(sequences)
                    loss = self.loss_function(outputs.squeeze(), labels)

                    total_val_loss += loss
            
            #Append the average training and validation loss for this epoch.
            ave_train_loss.append(total_loss/train_count)
            ave_val_loss.append(total_val_loss/val_count)
            
            #Print the training and validation loss after epoch_print number of epochs.
            if (epoch+1)%epoch_print == 0:
                print("Epoch %d finished. Train loss: %f. Validation loss: %f" % 
                     (epoch+1, ave_train_loss[-1], ave_val_loss[-1]))
        return {"train_loss":ave_train_loss, "val_loss":ave_val_loss}
    
    def get_lengths_and_labels(self, batch_data):
        """
        Get the lengths and the final results of games.
        
        Inputs:
            batch_data: torch.util.data.Dataset. Should be dataloader.PreprocessedParsedReplayDataset class.
        Returns:
            dictionary:
                        "lengths":torch.Tensor, size(batch_size). Lengths of the games in batch.
                        "results":numpy.ndarray, size(batch_size). Results of the games in the batch. -1 if dire winned and +1 if radiant winned.
        """
        end_labels = batch_data["labels"][:,-1].numpy()
        results = np.where(end_labels>0, 1.0, -1.0)
        return {"lengths":batch_data["lengths"], "results":results}

    def predict(self, batch_data, threshold=0.25):
        """
        Make predictions with a 30 second interval.
        
        Inputs:
            batch_data: torch.util.data.Dataset. Should be dataloader.PreprocessedParsedReplayDataset class.
        Returns:
            predictions: torch.Tensor, size=(batch_size, L). Predictions of the probability that radiant will win. batch_size should only be 1. L is the length of the game.
        """
        with torch.no_grad():
            dim = batch_data["features"].shape
            
            #If batch_size != 1
            if len(dim) == 3:
                self.model.hidden_cell = self.model.init_hidden(self.model.batch_size)
                predictions = self.model(batch_data["features"].to(self.model.device))
                
            #If batch_size == 1, append a new dimension of 1.
            else:
                self.model.hidden_cell = self.model.init_hidden(1)
                predictions = self.model(batch_data["features"].view(1,dim[0],dim[1]).to(self.model.device))
                
        #predictions is after tanh function, so should scale to (0,1) as the predicted probability of radiant win.
        return predictions / 2.0 + 0.5
    
class LSTMWithH2vTrain:
    """
    The class to train the LSTM baseline model and make predictions.
    """
    def __init__(self, input_dim, hidden_dim, h2v_dim=20, h2v_layer_dim=[50,30,1], 
                 output_dim=1, num_epochs=100, lr=0.01, loss_function=nn.MSELoss(), 
                 batch_size=10, device=torch.device('cpu'), **kwargs):
        """
        Inputs:
            input_dim, hidden_dim: arguments passing to model.LSTM_baseline class.
        Keywords:
            output_dim, batch_size, device: keywords passing to model.LSTM_baseline class.
            num_epoches: int. Number of epochs used for training.
            lr: float. Learning rate used for training.
            loss_function: torch.nn loss function. Loss function used for the training.
            **kwargs: keywords passing to dataloader.split_dataloader function.
        """
        self.dataloader = split_dataloader(PreprocessedParsedReplayDataset, batch_size=batch_size, **kwargs)
        self.model = LSTMWithH2vSubnet(input_dim, hidden_dim, h2v_dim=20, h2v_layer_dim=[50,30,1], 
                                      output_dim=output_dim, batch_size=batch_size, device=device)
        if self.model.device == torch.device('cuda'):
            self.model.cuda()
        self.num_epochs = num_epochs
        self.loss_function = loss_function
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        
    def train(self, epoch_print=10):
        """
        The function to train the model.
        
        Returns:
            dictionary:
                    "train_loss":list. Average training loss at each epoch.
                    "val_loss":list. Average validation loss at each epoch.
        Keywords:
            epoch_print: int. Number of epochs between printing out the train and validation loss.
        """
        #List of verage training and validation loss
        ave_train_loss = []
        ave_val_loss = []
        
        #For each epoch
        for epoch in range(self.num_epochs):
            
            #Training and validation loss for this single epoch.
            total_loss = 0
            total_val_loss = 0
            
            #Count the number of training data
            train_count = 0
            for batch in self.dataloader['train']:
                sequences, labels, embeddings = batch["features"].to(self.model.device), batch["labels"].to(self.model.device), batch["embeddings"].to(self.model.device)

                train_count += 1
                
                #Initialize the hidden layer and zero the stored gradient.
                self.model.hidden_cell = self.model.init_hidden(self.model.batch_size)
                self.model.zero_grad()

                outputs = self.model(sequences, embeddings).squeeze()
                loss = self.loss_function(outputs, labels)
                
                #Backprop and optimize.
                loss.backward()
                self.optimizer.step()

                total_loss += loss.cpu().data

            #Count the number of validation data
            val_count = 0
            
            #Do not store gradiant, since we are not training.
            with torch.no_grad():
                for batch in self.dataloader['val']:
                    
                    sequences, labels, embeddings = batch["features"].to(self.model.device), batch["labels"].to(self.model.device), batch["embeddings"].to(self.model.device)
                    val_count += 1
                    
                    self.model.hidden_cell = self.model.init_hidden(self.model.batch_size)

                    outputs = self.model(sequences, embeddings)
                    loss = self.loss_function(outputs.squeeze(), labels)

                    total_val_loss += loss.cpu().data
            
            #Append the average training and validation loss for this epoch.
            ave_train_loss.append(total_loss/train_count)
            ave_val_loss.append(total_val_loss/val_count)
            
            #Print the training and validation loss after epoch_print number of epochs.
            if (epoch+1)%epoch_print == 0:
                print("Epoch %d finished. Train loss: %f. Validation loss: %f" % 
                     (epoch+1, ave_train_loss[-1], ave_val_loss[-1]))
        return {"train_loss":ave_train_loss, "val_loss":ave_val_loss}
    
    def get_lengths_and_labels(self, batch_data):
        """
        Get the lengths and the final results of games.
        
        Inputs:
            batch_data: torch.util.data.Dataset. Should be dataloader.PreprocessedParsedReplayDataset class.
        Returns:
            dictionary:
                        "lengths":torch.Tensor, size(batch_size). Lengths of the games in batch.
                        "results":numpy.ndarray, size(batch_size). Results of the games in the batch. -1 if dire winned and +1 if radiant winned.
        """
        end_labels = batch_data["labels"][:,-1].numpy()
        results = np.where(end_labels>0, 1.0, -1.0)
        return {"lengths":batch_data["lengths"], "results":results}

    def predict(self, batch_data, threshold=0.25):
        """
        Make predictions with a 30 second interval.
        
        Inputs:
            batch_data: torch.util.data.Dataset. Should be dataloader.PreprocessedParsedReplayDataset class.
        Returns:
            predictions: torch.Tensor, size=(batch_size, L). Predictions of the probability that radiant will win. batch_size should only be 1. L is the length of the game.
        """
        with torch.no_grad():
            dim = batch_data["features"].shape
            
            #If batch_size != 1
            if len(dim) == 3:
                self.model.hidden_cell = self.model.init_hidden(self.model.batch_size)
                predictions = self.model(batch_data["features"].to(self.model.device), batch_data["embeddings"].to(self.model.device))
                
            #If batch_size == 1, append a new dimension of 1.
            else:
                self.model.hidden_cell = self.model.init_hidden(1)
                predictions = self.model(batch_data["features"].view(1,dim[0],dim[1]).to(self.model.device), batch_data["embeddings"].view(1,20).to(self.model.device))
                
        #predictions is after tanh function, so should scale to (0,1) as the predicted probability of radiant win.
        return predictions / 2.0 + 0.5
    
