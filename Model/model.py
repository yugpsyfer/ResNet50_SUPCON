import torch.nn as nn
import torchvision
import torchvision.models as models


class resNet50():
    def __init__(self):
        model_ = models.resnet50(pretrained=False)
