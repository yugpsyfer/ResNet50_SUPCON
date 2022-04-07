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
from Data.mini_imagenet import MiniImageNet
import matplotlib.pyplot as plt
import importlib
import logging
import time

log_path = './Outputs/Logs/'
logging.basicConfig(filename=log_path + 'Run.log', encoding='utf-8', level=logging.INFO)

output_model_path = "./Outputs/Pretrained_Models/"
criterion_options = dict()
criterion_options['SupCon'] = {"epochs": 1000,
                               "temperature": 0.5,
                               "annealing": "cosine",
                               "learning_rate": 0.5,
                               "optimizer": "sgd"}

criterion_options['CE'] = {"epochs": 500,
                           "optimizer": "sgd",
                           "learning_rate": 0.8}


def load_pretrained_model(model_name):
    load_path = os.path.join(output_model_path, model_name)
    model_ = torch.load(load_path)

    return model_


def make_model_(pretrain, criterion_loss, model_name):
    model_with_fc = resNet50()
    final_model = None
    layers = list(model_with_fc.model_.children())[:-1]
    if pretrain:
        if criterion_loss == "SupCon":
            model_with_fc.model_.fc = nn.Linear(in_features=2048, out_features=300, bias=True)
            final_model = model_with_fc.model_

        elif criterion_loss == "CE":
            model_with_fc.model_.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=100, bias=True),
                                                    nn.Softmax(dim=1))
            final_model = model_with_fc.model_
        else:
            raise ValueError
    else:
        pretrained_model = load_pretrained_model(model_name) #NEEDS EDITING
        final_model = torch.nn.Sequential(*(list(pretrained_model.model_.children())[:-1]))
        final_model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=100, bias=True),
                                       nn.Softmax(dim=1))
    return final_model


def get_cuda_device():
    test_tensor = torch.tensor(0)
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        for i in range(count):
            try:
                logging.info("Trying CUDA device " + str(i))
                device = torch.device('cuda:'+str(i))
                test_tensor = test_tensor.to(device)
                logging.info("Using CUDA device "+str(i))
                break
            except BaseException:
                logging.info("Cuda Device "+str(i)+" is busy. Trying other devices")
    else:
        raise EnvironmentError("CUDA NOT AVAILABLE")


    return device


def prepare_dataloader(dataset_class, batch_size, crit):
    dataset = dataset_class(root_dir="./Inputs/mini_image_net_merged/", label_file="./Inputs/Labels/wordnet_details.txt", criterion=crit)
    # val_size = int(len(dataset) * 0.1)
    # train_size = len(dataset) - int(len(dataset) * 0.1)
    # train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_ = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # val_ = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    return train_, None


def draw_and_save_plots(history_, ep):
    plot_path = "./Outputs/plots/"
    train = [i[1] for i in history_['train']]

    epo = np.linspace(start=0, stop=ep, num=50)

    plt.figure(figsize=(10, 10))

    plt.plot(epo, train, color='blue', label='Train LOSS VS EPOCHS', linewidth=2, linestyle='dashed')
    # plt.plot(epo, val, color='red', label='Validation LOSS VS EPOCHS', linewidth=2, linestyle='dashed')

    plt.xlabel("EPOCHS")
    plt.ylabel("LOSS")
    plt.legend()

    plt.savefig(plot_path+"output.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process options for training')
    parser.add_argument('--loss_criterion', type=str, default="SupCon",
                        choices=["SupCon", "CE"], help="Choose the loss criterion")
    parser.add_argument('--pretrain', type=bool, default=True, help="Choose if you want  to pretrain")
    parser.add_argument('--batch_size', type=int, default=1024, help="Choose batch size for training")
    parser.add_argument('--model_eval', type=bool, default=False, help="Choose whether to test model")
    parser.add_argument('--model_to_train', type=str, help="Choose th model to train")

    opt = parser.parse_args()
    dev = get_cuda_device()

    if opt.pretrain:
        if opt.loss_criterion == "SupCon":
            model = make_model_(opt.pretrain, opt.loss_criterion, None)
            temperature = criterion_options['SupCon']['temperature']
            base_temperature = temperature
            l_r = criterion_options['SupCon']['learning_rate']
            optimr = importlib.import_module("torch.optim." + criterion_options['SupCon']['optimizer'])
            optimizer = optimr.SGD(lr=l_r, params=model.parameters())
            epochs = criterion_options['SupCon']['epochs']
            criterion = ["SupCon", SupConLoss(temperature=temperature, contrast_mode='all',
                                              base_temperature=base_temperature, device=dev)]

        elif opt.loss_criterion == "CE":
            model = make_model_(opt.pretrain, opt.loss_criterion, None)
            epochs = criterion_options['CE']['epochs']
            l_r = criterion_options['CE']['learning_rate']
            optimr = importlib.import_module("torch.optim." + criterion_options['CE']['optimizer'])
            optimizer = optimr.SGD(lr=l_r, params=model.parameters())
            criterion = ["CE", F.cross_entropy]
        else:
            raise NameError("Loss not supported")
    else:
        pass

    model.to(dev)

    train_dl, val_dl = prepare_dataloader(MiniImageNet, opt.batch_size, opt.loss_criterion)
    logging.info("STARTING TRAINING==================")
    start_time = time()

    trained_model, history = training.train(train_dl=train_dl, val_dl=val_dl,
                                            criterion=criterion, epochs=epochs,
                                            optimizer=optimizer, dev=dev, model=model)

    end_time = time()
    seconds_elapsed = end_time - start_time
    hours, rest = divmod(seconds_elapsed, 3600)

    logging.info("TIME REQUIRED TO TRAIN THE MODEL:"+hours+" hrs")
    draw_and_save_plots(history_=history, ep=epochs)

    model_save_path = os.path.join(output_model_path, opt.loss_criterion + ".pth")
    torch.save(trained_model, model_save_path)



