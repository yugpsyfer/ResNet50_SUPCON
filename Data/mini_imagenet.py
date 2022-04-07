import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class MiniImageNet(Dataset):
    def __init__(self, root_dir, label_file, criterion):
        self.rootDir = root_dir
        self.allFiles = os.listdir(self.rootDir)
        self.label_dict = dict()
        self.criter = criterion

        if criterion == "CE":
            self.tensorTransformation = transforms.ToTensor()
            labels = pd.read_csv(label_file, delimiter=" ")
            labels = enumerate(labels['wdnet_id'].to_list())

            for k, v in labels:
                self.label_dict[v] = k
        else:
            mean = [0.4731, 0.4489, 0.4038]
            std = [0.2631, 0.2556, 0.2700]
            normalize = transforms.Normalize(mean=mean, std=std)
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
            self.tensorTransformation = TwoCropTransform(train_transform)
            embedding_path = "./Outputs/Knowledge_Graphs/embeddings.npy"
            self.embeddings = np.load(embedding_path, allow_pickle=True)
            self.embeddings = self.embeddings[()]

            labels = pd.read_csv(label_file, delimiter=" ")
            labels = enumerate(labels['wdnet_id'].to_list())

            for k, v in labels:
                self.label_dict[v] = k

    def __len__(self):
        return len(os.listdir(self.rootDir))

    def __getitem__(self, index):
        if self.criter == 'CE':
            return self._ce_get_item(index)
        else:
            return self._supcon_get_item(index)

    def _ce_get_item(self, index):
        files = self.allFiles[index]
        file_name = files.split("_")[0]
        label = torch.tensor(data=self.label_dict[file_name])

        with Image.open(self.rootDir + files) as im:
            z = im.resize((32, 32))
            z = np.array(z)
            data = self.tensorTransformation(z)

        return data, label  # Return a tuple format (data,label)

    def _supcon_get_item(self, index):
        files = self.allFiles[index]
        file_name = files.split("_")[0]
        label = torch.tensor(data=self.label_dict[file_name])
        _embedding_ = torch.tensor(data=[self.embeddings[file_name].flatten()], dtype=torch.double)
        embedding = torch.cat((_embedding_, _embedding_), dim=0)
        print(embedding.shape)
        with Image.open(self.rootDir + files) as im:
            data = self.tensorTransformation(im)

        # data = torch.Tensor(data)
        # embedding = torch.Tensor(embedding)

        return (data, label, embedding)  # Return a tuple format (data,label)
