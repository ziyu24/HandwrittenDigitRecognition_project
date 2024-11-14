"""
Created on 11 13, 2024
@author: <Cui>
@bref: 创建 DataLoader
"""

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from project.src.common.config import config_yaml


class DataLoaderMNIST:
    def __init__(self, data_path, batch_size, shuffle=True, download=config_yaml['data']['download']):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.download = download

    def get_data_loader(self, is_train):
        to_tensor = transforms.Compose([transforms.ToTensor()])
        data_set = MNIST(self.data_path, is_train, transform=to_tensor, download=self.download)
        return DataLoader(data_set, self.batch_size, self.shuffle)
