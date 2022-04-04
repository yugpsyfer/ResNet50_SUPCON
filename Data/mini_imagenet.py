import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd


class MiniImageNet(Dataset):
    def __init__(self, root_dir, label_file, criterion):
        self.rootDir = root_dir
        self.tensorTransformation = transforms.ToTensor()
        self.allFiles = os.listdir(self.rootDir)
        self.label_dict = dict()

        if criterion == "CE":
            labels = pd.read_csv(label_file, delimiter=" ")
            labels = enumerate(labels['wdnet_id'].to_list())

            for k, v in labels:
                self.label_dict[v] = k
        else:
            embedding_path = "./Outputs/Knowledge_Graphs/embeddings.npy"
            self.label_dict = np.load(embedding_path, allow_pickle=True)

    def __len__(self):
        return len(os.listdir(self.rootDir))

    def __getitem__(self, index):
        files = self.allFiles[index]
        file_name = files.split("_")[0]
        label = torch.tensor(data=self.label_dict[file_name], dtype=float)

        with Image.open(self.rootDir + files) as im:
            z = im.resize((32, 32))
            z = np.array(z)
            data = self.tensorTransformation(z)

        return data, label  # Return a tuple format (data,label)