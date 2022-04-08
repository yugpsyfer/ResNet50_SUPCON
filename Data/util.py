import torch
from torch.utils.data.dataloader import DataLoader
from mini_imagenet import MiniImageNet

import os
import pandas as pd
import json
import shutil


def make_imagnet_V2():
    wordnet_details = '../Inputs/Labels/wordnet_details.txt'
    imagenet_v2_meta = '../Inputs/Labels/imagenet_v2_class_map.json'
    target_dataset = '../Inputs/imagenetV2/'
    output = '../Outputs/Target/'
    # /os.mkdir(output + 'imagenet_v2')

    dataset_sub_folders = os.listdir(target_dataset)
    labels = pd.read_csv(wordnet_details, delimiter=" ")
    classes_ = labels['wdnet_id'].to_list()

    mapper_ =dict()

    for lab_ in classes_:
        mapper_[lab_] = 0

    with open(imagenet_v2_meta, 'r') as fp:
        meta = json.load(fp)
    counter = 0

    for files in meta:
        if mapper_.get(files['wnid'], -1) >= 0:
            path__ = target_dataset+str(files['cid'])+"/"
            images = os.listdir(path__)

            for image in images:
                src = path__ + str(image)
                dst = output + 'imagenet_v2/' + files['wnid'] + "_" + str(counter)
                shutil.copy(src, dst)
                counter += 1
                mapper_[files['wnid']] += 1

    with open(output+"target_stats.json", 'w') as fp:
        json.dump(mapper_, fp)



def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == "__main__":
    # dataset = MiniImageNet(root_dir="../Inputs/mini_image_net_merged/",
    #                         label_file="../Inputs/Labels/wordnet_details.txt", criterion="CE")
    # dataLoa = DataLoader(dataset, batch_size=64)

    # print(get_mean_and_std(dataLoa))
    make_imagnet_V2()
