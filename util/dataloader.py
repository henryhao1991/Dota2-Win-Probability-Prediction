# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries
import os
import numpy as np
from sklearn.model_selection import train_test_split


class PreprocessedParsedReplayDataset(Dataset):
    """
    Customed preprocessed input files for parsed game replays.
    Features/labels should be in folder ./data/feautures and ./data/labels respectively, unless specified.
    Each txt file in the folder is for 1 game.
    """
    
    def __init__(self, feature_folder='./data/mixed/features/', label_folder='./data/mixed/labels/', embedding_folder='./data/mixed/embedding_feature/'):
        """
        Keywords:
            feature_folder: string. The folder path for input features.
            label_folder: string. The folder path for labels.
        """
        self.feature_folder = feature_folder
        self.label_folder = label_folder
        self.embedding_folder = embedding_folder
        assert len(os.listdir(self.feature_folder))==len(os.listdir(self.label_folder))==len(os.listdir(self.embedding_folder)), "Number of files in feature, label and embedding folders do not match."
        self.num_games = len(os.listdir(self.feature_folder))
        
    def __len__(self):
        return self.num_games
    
    def __getitem__(self, ind):
        """
        Inputs:
            ind: int. The index of the data.
        Returns:
            dictionary:
                        "features":torch.Tensor, size=(*,32) * is the length of the game. Input features of the model.
                        "labels":torch.Tensor, size=(*) * is the length of the game. Label of the model.
                        "lengths":int. Length of the game, used for prediction and generating probability graph of a single game.
        """
        feature_path = os.path.join(self.feature_folder,str(int(ind))+'.txt')
        label_path = os.path.join(self.label_folder,str(int(ind))+'.txt')
        embedding_path = os.path.join(self.embedding_folder,str(int(ind))+'.txt')
        
        #To do: check if file exists before using np.loadtxt
        features = np.loadtxt(feature_path)
        labels = np.loadtxt(label_path)
        length = len(labels)
        embeddings = np.loadtxt(embedding_path)
        
        #Return features/labels/lengths_of_game as a dictionary
        return {"features":torch.Tensor(features), "labels":torch.Tensor(labels), "embeddings":torch.Tensor(embeddings), "lengths":length}
    
    
class Hero2vecDataset(Dataset):
    '''
    The dataloader class to load data for the hero2vec training.
    Inspired by the following repo:
    https://github.com/ybw9000/hero2vec
    '''
    
    def __init__(self, lineup_file = './data/mixed_lineup.txt'):
        assert os.path.exists(lineup_file), "Lineup file {} doesn't exist.".format(lineup_file)
        self.lineup = np.loadtxt(lineup_file) - 1
        
    def __len__(self):
        return len(self.lineup) * 5
    
    def __getitem__(self, ind):
        '''
        Inputs:
            ind: int
        Returns:
            dictionary:
                        "context": torch.LongTensor, size = (4,). The 4 context heroes within the team.
                        "target": int. The target hero
        '''
        game_ind, hero_ind = divmod(ind,5)
        context_heroes = np.concatenate((self.lineup[game_ind, :hero_ind], self.lineup[game_ind, hero_ind+1:]))
        target = self.lineup[game_ind, hero_ind]

        return {"context":torch.LongTensor(context_heroes), "target":int(target)}


def PadSequence(batch):
    '''
    The function to pad sequences in a batch with 0.
    It will be passed to DataLoader as the keyword argument of collate_fn.
    '''
    
    #Get the lengths but don't do anything
    lengths = [x["lengths"] for x in batch]
    max_length = max(lengths)
    lengths = torch.Tensor(lengths)
    
    #Get the embedding feature but don't do anything
    embeddings = [x["embeddings"] for x in batch]
    
    #Get the sequences, i.e., features and pad them
    sequences = [x["features"] for x in batch]
    sequences_padded = []
    for s in sequences:
        single_game_sequence = s
        while single_game_sequence.shape[0] < max_length:
            single_game_sequence = torch.cat((single_game_sequence, single_game_sequence[-1,:].view(1,-1)))
        sequences_padded.append(single_game_sequence)
    
    #Get the labels and pad them
    labels = [x["labels"] for x in batch]
    labels_padded = []
    for l in labels:
        single_game_label = l
        while single_game_label.shape[0] < max_length:
            single_game_label = torch.cat((single_game_label, single_game_label[-1].view(1)))
        labels_padded.append(single_game_label)
    
    #return the padded_features/padded_labels/lengths_of_game and labels as a dictionary
    return {"features":torch.stack(sequences_padded), "labels":torch.stack(labels_padded), "embeddings":torch.stack(embeddings), "lengths":lengths}


def split_dataloader(dataset, batch_size=1, total_num_games=-1, p_val=0.1, p_test=0.2, seed=3154, shuffle=True, 
                    collate_fn=None, **kwargs):
    '''
    The function to split the dataset into training/validation/test sets.
    Inputs:
        dataset: torch.util.data.Dataset. Dataset used for the model.
    Returns:
        dictionary.
                    "train":torch.utils.data.DataLoader. Dataloader for training set.
                    "val":torch.utils.data.DataLoader. Dataloader for validation set.
                    "test":torch.utils.data.DataLoader Dataloader for test set.
    Keywords:
        batch_size: int. The batch size for the model inputs.
        total_num_games: int. Total number of games used. if -1, then all the data in the dataset will be used.
        p_val: float. Percentage of data used for validation set. if 0 return None as validation dataset.
        p_test: float. Percentage of data used for test set. if 0, return None as test dataset.
        seed: int. Random seed used.
        shuffle: bool. If shuffle the dataset or not.
        collate_fn: the collate function passed into the dataloader. for more information, check the PyTorch documentation of Dataloader class.
        **kwargs: the keywords passing to the dataset class.
    '''
    
    #Load the dataset from files in the respective folders
    dataset = dataset(**kwargs)
    
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
        
        #Check if collate_fn == None. Somehow pass None as collate_fn gives an error.
        if collate_fn:
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_val, collate_fn=collate_fn)
        else:
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_val)
            
    else:
        train_ind = all_ind[:]
        val_loader = None
        
    #Check if the test set is needed. I.e., p_test > 0
    if p_test > 0:
        train_ind, test_ind = train_test_split(train_ind, test_size=p_test, random_state=seed, shuffle=shuffle)
        sample_test = SubsetRandomSampler(test_ind)
        
        #Check if collate_fn == None. Somehow pass None as collate_fn gives an error.
        if collate_fn:
            test_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_test, collate_fn=collate_fn)
        else:
            test_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_test)
            
    else:
        test_loader = None
    
    sample_train = SubsetRandomSampler(train_ind)
    
    #Check if collate_fn == None. Somehow pass None as collate_fn gives an error.
    if collate_fn:
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_train, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_train)
    
    #Return the dictionary of all the dataloaders
    return {"train":train_loader, "val":val_loader, "test":test_loader}


