import torch
from torch.utils.data.dataloader import DataLoader
from mini_imagenet import MiniImageNet


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
    dataset = MiniImageNet(root_dir="../Inputs/mini_image_net_merged/",
                            label_file="../Inputs/Labels/wordnet_details.txt", criterion="CE")
    dataLoa = DataLoader(dataset, batch_size=64, shuffle=True)

    print(get_mean_and_std(dataLoa))
