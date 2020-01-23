# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries for data manipulation and visualization
import os
import numpy as np


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
        
        features = np.loadtxt(feature_path)
        labels = np.loadtxt(label_path)
        
        return (features, labels)


def split_dataloader(total_num_games=-1, p_val=0.1, p_test=0.2, seed=3154, shuffle=True):
    
    dataset = PreprocessedParsedReplayDataset()
    
    dataset_size = len(dataset)
    all_ind = list(range(dataset_size))
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_ind)
        
    if total_num_games == -1:
        last_ind = dataset_size
    else:
        assert total_num_games > 0 and total_num_games <= dataset_size, "Invalid total number of games. Has to be > 0 and < dataset size"
        last_ind = total_num_games
        
    val_split = int(np.floor(p_val * last_ind))
    train_ind, val_ind = all_ind[val_split:last_ind], all_ind[:val_split]
    
    test_split = int(np.floor(p_test * len(train_ind)))
    train_ind, test_ind = all_ind[test_split:last_ind], all_ind[:test_split]
    
    sample_train = SubsetRandomSampler(train_ind)
    sample_val = SubsetRandomSampler(val_ind)
    sample_test = SubsetRandomSampler(test_ind)
    
    train_loader = DataLoader(dataset, sampler=sample_train)
    val_loader = DataLoader(dataset, sampler=sample_val)
    test_loader = DataLoader(dataset, sampler=sample_test)
    
    return (train_loader, val_loader, test_loader)

def single_dataloader(total_num_games=-1, seed=3154, shuffle=False):
    
    dataset = PreprocessedParsedReplayDataset()
    
    dataset_size = len(dataset)
    all_ind = list(range(dataset_size))
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_ind)
        
    if total_num_games == -1:
        last_ind = dataset_size
    else:
        assert total_num_games > 0 and total_num_games <= dataset_size, "Invalid total number of games. Has to be > 0 and < dataset size"
        last_ind = total_num_games
    all_loader = DataLoader(dataset)
    
    return all_loader
