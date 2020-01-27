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
        
        features = np.loadtxt(feature_path)
        labels = np.loadtxt(label_path)
        
        return {"features":torch.Tensor(features), "labels":torch.Tensor(labels)}


def PadSequence(batch):
        #Each element in batch is (features, labels)
    sequences = [x["features"] for x in batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

    labels = [x["labels"] for x in batch]
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return {"features":sequences_padded, "labels":labels_padded}


def split_dataloader(batch_size=1, total_num_games=-1, p_val=0.1, p_test=0.2, seed=3154, shuffle=True, 
                    feature_folder='./data/features/', label_folder='./data/labels/'):
    
    dataset = PreprocessedParsedReplayDataset(feature_folder, label_folder)
    
    if total_num_games == -1:
        dataset_size = len(dataset)
    else:
        assert total_num_games > 0 and total_num_games <= len(dataset), "Invalid total number of games. Has to be > 0 and < dataset size"
        dataset_size = total_num_games
        
    all_ind = list(range(dataset_size))
    
    if p_val > 0:
        train_ind, val_ind = train_test_split(all_ind, test_size=p_val, random_state=seed, shuffle=shuffle)
        sample_val = SubsetRandomSampler(val_ind)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_val, collate_fn=PadSequence)
    else:
        train_ind = all_ind[:]
        val_loader = None
        
    if p_test > 0:
        train_ind, test_ind = train_test_split(train_ind, test_size=p_test, random_state=seed, shuffle=shuffle)
        sample_test = SubsetRandomSampler(test_ind)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_test, collate_fn=PadSequence)
    else:
        test_loader = None
    
    sample_train = SubsetRandomSampler(train_ind)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_train, collate_fn=PadSequence)
    
    return (train_loader, val_loader, test_loader)

'''
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
'''
