import numpy as np
import argparse
import os
from tqdm import tqdm
import time
import pickle as pkl
import torch

class Dataloader(object):
    def __init__(self, dataset_tuple, batch_size, shuffle) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_tuple = dataset_tuple
        self.dataset_tuple = [torch.from_numpy(t) for t in self.dataset_tuple]
        
        self.dataset_size = len(self.dataset_tuple[0])
        if self.dataset_size % self.batch_size == 0:
            self.total_step = int(self.dataset_size / self.batch_size)
        else:
            self.total_step = int(self.dataset_size / self.batch_size) + 1
        self.step = 0
    
    def __len__(self):
        return self.total_step

    def _shuffle_data(self):
        print('shuffling...')
        perm = np.random.permutation(self.dataset_size)
        for i in range(len(self.dataset_tuple)):
            self.dataset_tuple[i] = self.dataset_tuple[i][perm]

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.total_step:
            raise StopIteration

        left = self.batch_size * self.step
        if self.step == self.total_step - 1:
            right = self.dataset_size
        else:
            right = self.batch_size * (self.step + 1)
        
        self.step += 1
        batch_data = []
        for i in range(len(self.dataset_tuple)):
            batch_data.append(self.dataset_tuple[i][left:right])
        return batch_data

    def refresh(self):
        print('refreshing...')
        self.step = 0
        if self.shuffle:
            self._shuffle_data()
        print('refreshed')


class DomainDataloader(object):
    def __init__(self, dataset_tuple, batch_size, shuffle) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_tuple = dataset_tuple
        x_domain = dataset_tuple[0][:,-1]
        unique_values = np.unique(x_domain)
        self.num_splits = len(unique_values)
        self.sub_datasets = []
        for value in unique_values:
            indexs = (x_domain == value)
            self.sub_datasets.append([data[indexs] for data in dataset_tuple]) 
            

        self.dataloaders = [Dataloader(self.sub_datasets[i],batch_size,shuffle) for i in range(len(self.sub_datasets))]

        
        self.dataset_size = len(self.dataset_tuple[0])
        self.total_step = 0
        for i in range(self.num_splits):
            self.total_step += len(self.dataloaders[i])
        self.step = 0
        self.num = 0
    
    def __len__(self):
        
        return self.total_step

    def _shuffle_data(self):
        print('shuffling...')

        for i in range(self.num_splits):
            self.dataloaders[i].refresh()

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.total_step:
            raise StopIteration

        batch_data = self.dataloaders[self.num].__next__()
        if self.dataloaders[self.num].step == self.dataloaders[self.num].total_step:
            self.num+=1
        self.step +=1
        
        return batch_data

    def refresh(self):
        print('refreshing...')
        self.step = 0
        self.num = 0
        for i in range(self.num_splits):
            self.dataloaders[i].step = 0
        if self.shuffle:
            self._shuffle_data()
        print('refreshed')



if __name__ == "__main__":
    with open('../data/Aliccp/feateng_data/input_data/train_set2.pkl', 'rb') as f:
        dataset_tuple = pkl.load(f)
    dl = DomainDataloader(dataset_tuple, 1024, True)
    for batch in tqdm(dl):
        train_x, train_y,train_x_theme_hist,train_x_hist = batch
        print(train_x)
        break