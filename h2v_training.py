#PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim

#Custom Python files
from util.dataloader import Hero2vecDataset, split_dataloader
from model.h2v_model import *

#Other libraries
import numpy as np
    

class Hero2vecTrain:
    '''
    Class for hero2vec training.
    '''
    
    def __init__(self, embedding_dim, heropool_size=129, lineup_file = './data/mixed_lineup.txt', 
                loss_function=nn.CrossEntropyLoss(), init_lr=0.1, epochs=100, lr_decay_epoch = 30,
              lr_decay_rate = 0.1, print_epoch = 10, gpu=False):
        """
        Inputs:
            embedding_dim: int. Size of the embedding layer.
        Keywords:
            heropool_size: int. The number of largest hero index. (Not the number of total heroes, since some index are not used currently in Dota2.)
            lineup_file: string. The file path of the lineup information.
            loss_function: torch.nn loss function. Loss function used for the training.
            init_lr: float. The initial learning rate.
            epochs: int. Total number of epochs for training.
            lr_decay_epoch: int. Number of epochs between learning rate decay.
            lr_decay_rate: float. The decay rate of the learning rate.
            print_epoch: int. Number of epochs between printing out the loss.
            gpu: bool. If GPU is used or not when available.
        """
        self.model = Hero2vecNetwork(embedding_dim, heropool_size=heropool_size)
        self.dataloader = split_dataloader(Hero2vecDataset, batch_size=10, lineup_file=lineup_file)
        self.loss_function = loss_function
        self.lr = init_lr
        self.epochs = epochs
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_rate = lr_decay_rate
        self.print_epoch = print_epoch
        self.gpu = gpu and torch.cuda.is_available()
        if self.gpu:
            model.cuda()
        
    def update_lr(self):
        self.lr = self.lr / self.lr_decay_rate
        
    def train(self):
        
        #List of verage training and validation loss
        loss_epochs = []
        val_loss_epochs = []
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            
            if epoch>0 and epoch%self.lr_decay_epoch == 0:
                self.update_lr()
                
            #Training and validation loss for this single epoch.
            total_loss = 0
            total_val_loss = 0
            
            for data in self.dataloader['train']:
                context, target = data['context'], data['target']
                
                #move data to gpu if it is used.
                if self.gpu:
                    context = context.cuda()
                    target = target.cuda()
                
                #zero the gradient stored.
                self.model.zero_grad()
                
                output = self.model(context)
                
                loss = self.loss_function(output, target)
                
                #Backprop and optimize.
                loss.backward()
                optimizer.step()
                
                total_loss += loss.cpu().data
                
            total_loss = total_loss/len(self.dataloader['train'])
            
            with torch.no_grad():
                for data in self.dataloader['val']:
                    context, target = data['context'], data['target']
                    
                    #move data to gpu if it is used.
                    if self.gpu:
                        context = context.cuda()
                        target = target.cuda()
                    
                    output = self.model(context)
                    
                    loss = self.loss_function(output, target)
                    
                    total_val_loss += loss.cpu().data
                    
                total_val_loss = total_val_loss/len(self.dataloader['val'])
            
            #Print the loss every print_epoch epochs
            if epoch%self.print_epoch == 0:
                print('epoch: %d, loss: %.3f, validation loss: %.3f' % (epoch, total_loss, total_val_loss))
                
            loss_epochs.append(total_loss)
            val_loss_epochs.append(total_val_loss)
            
        return {'train_loss':loss_epochs, 'val_loss':val_loss_epochs}
    
    def save_model(self, file_dir):
        
        #To do: check if the folder exists.
        torch.save(self.model, file_dir)
        
    def save_embeddings(self, file_dir):
        
        #To do: check if the folder exists.
        embeddings = self.model.hero_embedding.weight.cpu().data.numpy()
        np.savetxt(file_dir, embeddings)
