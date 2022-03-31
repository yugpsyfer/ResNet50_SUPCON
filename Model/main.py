import torch
import torch.optim as optim
from Model.model import resNet50
import training
#from ..Data import

if __name__ == __main__:
    model = resNet50()
    model = model.model_

    dev = torch.device('cuda:0')
    model.to_device(dev)

    optim = optim.SGD(params=model.params, lr=0.5)

    training.train()