import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

class MiniImageNet(Dataset):
    def __init__(self, rootDir, labelFile):
        self.rootDir = rootDir
        self.tensorTransformation = transforms.ToTensor()
        self.allFiles = os.listdir(self.rootDir)
        self.labels =

    def __len__(self):
        return len(os.listdir(self.rootDir))

    def __getitem__(self, index):
        files = self.allFiles[index]
        file_name = files.split("_")[0]
        label = torch.tensor(data=self.labels[file_name], dtype=int)

        with Image.open(self.rootDir + files) as im:
            z = im.resize((32, 32))
            z = np.array(z)
            data = self.tensorTransformation(z)

        return data, label  # Return a tuple format (data,label)