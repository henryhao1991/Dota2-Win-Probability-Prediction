# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries for data manipulation and visualization
import os
import numpy as np
from sklearn.model_selection import train_test_split


class PreprocessedParsedReplayDataset(Dataset):
    """
    Customed preprocessed input files for parsed game replays.
    Features/labels should be in folder ./data/feautures and ./data/labels respectively, unless specified.
    Each txt file in the folder is for 1 game.
    """
    
    def __init__(self, feature_folder='./data/features/', label_folder='./data/labels/'):
        
        self.feature_folder = feature_folder
        self.label_folder = label_folder
        assert len(os.listdir(self.feature_folder))==len(os.listdir(self.label_folder)), "Number of files in feature and label folders do not match."
        self.num_games = len(os.listdir(self.feature_folder))
        
    def __len__(self):
        return self.num_games
    
    def __getitem__(self, ind):
        
        feature_path = os.path.join(self.feature_folder,str(int(ind))+'.txt')
        label_path = os.path.join(self.label_folder,str(int(ind))+'.txt')
        
        #To do: check if file exists before using np.loadtxt
        features = np.loadtxt(feature_path)
        labels = np.loadtxt(label_path)
        
        return {"features":torch.Tensor(features), "labels":torch.Tensor(labels)}


def PadSequence(batch):
    '''
    The function to pad sequences in a batch with 0.
    It will be passed to DataLoader as the keyword argument of collate_fn.
    '''
    
    #Get the sequences, i.e., features and pad them
    sequences = [x["features"] for x in batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    
    #Get the labels and pad them
    labels = [x["labels"] for x in batch]
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    #return the sequences/features and labels as a dictionary
    return {"features":sequences_padded, "labels":labels_padded}


def split_dataloader(batch_size=1, total_num_games=-1, p_val=0.1, p_test=0.2, seed=3154, shuffle=True, 
                    feature_folder='./data/features/', label_folder='./data/labels/'):
    '''
    The function to split the dataset into training/validation/test sets.
    If total_num_games == -1, all data will be used. Otherwise will only use the first total_num_games number of data.
    p_val and p_test can be 0. In that case, the val_loader and test_loader will be None respectively.
    '''
    
    #Load the dataset from files in the respective folders
    dataset = PreprocessedParsedReplayDataset(feature_folder, label_folder)
    
    #Check total_num_games and get the desired dataset size.
    if total_num_games == -1:
        dataset_size = len(dataset)
    else:
        assert total_num_games > 0 and total_num_games <= len(dataset), "Invalid total number of games. Has to be > 0 and < dataset size"
        dataset_size = total_num_games
        
    all_ind = list(range(dataset_size))
    
    #Check if the validation set is needed. I.e., p_val > 0
    if p_val > 0:
        train_ind, val_ind = train_test_split(all_ind, test_size=p_val, random_state=seed, shuffle=shuffle)
        sample_val = SubsetRandomSampler(val_ind)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_val, collate_fn=PadSequence)
    else:
        train_ind = all_ind[:]
        val_loader = None
        
    #Check if the test set is needed. I.e., p_test > 0
    if p_test > 0:
        train_ind, test_ind = train_test_split(train_ind, test_size=p_test, random_state=seed, shuffle=shuffle)
        sample_test = SubsetRandomSampler(test_ind)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_test, collate_fn=PadSequence)
    else:
        test_loader = None
    
    sample_train = SubsetRandomSampler(train_ind)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_train, collate_fn=PadSequence)
    
    #Return the dictionary of all the dataloaders
    return {"train":train_loader, "val":val_loader, "test":test_loader}


