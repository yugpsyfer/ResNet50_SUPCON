import argparse
import os.path
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from Model.model import resNet50
from Model.supconloss import SupConLoss
from Model import training
from Data import mini_imagenet
import matplotlib.pyplot as plt


output_model_path = "./Outputs/Pretrained_Models/"
criterion_options = dict()
criterion_options['SupCon'] = {"epochs": 1000,
                               "temperature": 0.5,
                               "annealing": "cosine",
                               "learning_rate": 0.5,
                               "optimizer": "SGD"}

criterion_options['CE'] = {"epochs": 500,
                           "optimizer": "SGD",
                           "learning_rate": 0.8}


def load_pretrained_model(model_name):
    load_path = os.path.join(output_model_path, model_name)
    model_ = torch.load(load_path)

    return model_


def make_model_(pretrain, criterion_loss, model_name):
    model_with_fc = resNet50()
    final_model = None
    if pretrain:
        if criterion_loss == "SupCon":
            final_model = torch.nn.Sequential(*(list(model_with_fc.model_.children())[:-1]))
            final_model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=300, bias=True))

        elif criterion_loss == "CE":
            final_model = torch.nn.Sequential(*(list(model_with_fc.model_.children())[:-1]))
            final_model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=100, bias=True),
                                           nn.Softmax())
        else:
            raise ValueError
    else:
        pretrained_model = load_pretrained_model(model_name) #NEEDS EDITING
        final_model = torch.nn.Sequential(*(list(pretrained_model.model_.children())[:-1]))
        final_model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=100, bias=True),
                                       nn.Softmax())
    return final_model


def get_cuda_device():
    test_tensor = torch.tensor(0)
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        for i in range(count):
            try:
                device = torch.device('cuda:'+str(i))
                test_tensor = test_tensor.to_device(device)
                print("Using CUDA device "+str(i))
                break
            except BaseException:
                print("Cuda Device "+str(i)+" is busy. Trying other devices")
    else:
        raise EnvironmentError("CUDA NOT AVAILABLE")

    return device


def prepare_dataloader(dataset, batch_size):
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - int(len(dataset) * 0.1)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_ = DataLoader(train_set, batch_size=batch_size)
    val_ = DataLoader(val_set, batch_size=batch_size)

    return train_, val_


def draw_and_save_plots(history_, ep):
    plot_path = "./Outputs/plots/"
    train = [i for i in history_['train']]
    val = [i[0] for i in history_['val']]

    epo = np.linspace(start=0, stop=ep, num=50)

    plt.figure(figsize=(10, 10))

    plt.plot(epo, train, color='blue', label='Train LOSS VS EPOCHS', linewidth=2, linestyle='dashed')
    plt.plot(epo, val, color='red', label='Validation LOSS VS EPOCHS', linewidth=2, linestyle='dashed')

    plt.xlabel("EPOCHS")
    plt.ylabel("LOSS")
    plt.legend()

    plt.savefig(plot_path+"output.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process options for training')
    parser.add_argument('--loss_criterion', type=str, default="SupCon",
                        choices=["SupCon,CE"], help="Choose the loss criterion")
    parser.add_argument('--pretrain', type=bool, default=True, help="Choose if you want  to pretrain")
    parser.add_argument('--batch_size', type=int, default=1024, help="Choose batch size for training")
    parser.add_argument('--model_eval', type=bool, default=False, help="Choose whether to test model")
    parser.add_argument('--model_to_train', type=str, help="Choose th model to train")

    opt = parser.parse_args()
    dev = get_cuda_device()

    if opt.pretrain:
        if opt.loss_criterion == "SupCon":
            temperature = criterion_options['SupCon']['temperature']
            base_temperature = temperature
            optimizer = optim.criterion_options['SupCon']['optimizer'](lr = criterion_options['SupCon']['learning_rate'])
            epochs = criterion_options['SupCon']['epochs']
            criterion = ["SupCon", SupConLoss(temperature=temperature, contrast_mode='all',
                                              base_temperature=base_temperature, device=dev)]

        elif opt.loss_criterion == "CE":
            epochs = criterion_options['CE']['epochs']
            optimizer = optim.criterion_options['CE']['optimizer'](lr=criterion_options['CE']['learning_rate'])
            criterion = ["CE", F.cross_entropy]
        else:
            raise NameError("Loss not supported")
    else:
        pass

    model = make_model_(opt.pretrain, opt.loss_criterion)
    model.to_device(dev)

    train_dl, val_dl = prepare_dataloader(mini_imagenet, opt.batch_size)

    trained_model, history = training.train(train_dl=train_dl, val_dl=val_dl,
                                            criterion=criterion, epochs=epochs,
                                            optimizer=optimizer, dev=dev, model=model)

    draw_and_save_plots(history_=history, ep=epochs)

    model_save_path = os.path.join(output_model_path, opt.loss_criterion + ".pth")
    torch.save(trained_model, model_save_path)



