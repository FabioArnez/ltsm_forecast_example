from typing import Optional, Any, Callable
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset 
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


class AirlineDataset(Dataset):
    def __init__(self,
                 dataset_file_path: str = 'airline-passengers.csv',
                 seq_length: int = 4,
                 ) -> None:
        super().__init__()

        self.dataset_file_path = dataset_file_path

        self.data_pd = pd.read_csv(self.dataset_file_path)
        self.data = self.data_pd.iloc[:, 1:2].values
        
        self.scaler = MinMaxScaler()
        self.data_scaled = self.scaler.fit_transform(self.data)

        self.seq_length = seq_length

        self.samples, self.labels = sliding_windows(self.data_scaled, self.seq_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = torch.Tensor(self.samples[index])
        label = torch.Tensor(self.labels[index])
        return sample, label


class AirlineDataModule(LightningDataModule):
    def __init__(self,
                 dataset_file_path: str = 'airline-passengers.csv',
                 train_size: float = 0.6,
                 batch_size: int =  30,
                 pin_memory: bool = True,
                 drop_last: bool = True,
                 num_workers: int =  10):
        super().__init__()
        
        self.dataset_file_path = dataset_file_path
        self.train_size = train_size
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.indices_train = None
        self.indices_test = None

    def setup(self, stage: Optional[str] = None) -> None:
        # create an instance of the dataset
        self.airline_dataset = AirlineDataset()
        # compute train & test sizes
        train_size = int(len(self.airline_dataset.labels) * self.train_size)
        valid_size = len(self.airline_dataset.labels) - train_size
        # get train and test data indices
        indices = list(range(len(self.airline_dataset)))
        self.indices_train = indices[0: train_size]
        self.indices_valid = indices[train_size: len(self.airline_dataset)]
        
        
    def train_dataloader(self) -> DataLoader:
        airline_data_train_loader = DataLoader(self.airline_dataset,
                                               batch_size=self.batch_size,
                                               sampler=self.indices_train,
                                               num_workers=self.num_workers,
                                               pin_memory=self.pin_memory,
                                               drop_last=self.drop_last,
                                               shuffle=False)
        
        return airline_data_train_loader
    
    def val_dataloader(self) -> DataLoader:
        airline_data_valid_loader = DataLoader(self.airline_dataset,
                                               batch_size=self.batch_size,
                                               sampler=self.indices_valid,
                                               num_workers=self.num_workers,
                                               pin_memory=self.pin_memory,
                                               drop_last=self.drop_last,
                                               shuffle=False)
        
        return airline_data_valid_loader

