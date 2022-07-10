import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd


class ImageNetV2(Dataset):
    def __init__(self, root_dir, label_file, criterion=None):
        self.rootDir = root_dir
        self.allFiles = os.listdir(self.rootDir)
        self.label_dict = dict()
        self.tensorTransformation = transforms.ToTensor()

        labels = pd.read_csv(label_file, delimiter=" ")
        labels = enumerate(labels['wdnet_id'].to_list())

        for k, v in labels:
            self.label_dict[v] = k

    def __len__(self):
        return len(os.listdir(self.rootDir))

    def __getitem__(self, index):
        files = self.allFiles[index]
        file_name = files.split("_")[0]
        label = torch.tensor(data=self.label_dict[file_name])

        with Image.open(self.rootDir + files) as im:
            z = im.resize((32, 32))
            z = np.array(z)
            data = self.tensorTransformation(z)
        data = data.to(torch.double)
        return (data, label)
