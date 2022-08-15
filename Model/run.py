"""
EACH FUNCTIONS NAME IS ACCORDING TO ITS FUNCTIONALITY
"""

import os.path
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from Model.supconloss import SupConLoss
from .training import train, test
from Data.mini_imagenet import MiniImageNet
from Data.target_dataset import ImageNetV2
from torch import optim
import logging
from time import time
from .model import ResNet


source_dataset_path = "./Inputs/mini_imagenet_subset/"  #CHANGED
target_dataset_path = "./Inputs/imagenet_v2_subset/"
output_model_path = "Outputs/Models/"


def get_cuda_device():
    test_tensor = torch.tensor(0)
    device = None
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        for i in range(count):
            try:
                logging.info("Trying CUDA device " + str(i))
                print("Trying CUDA device " + str(i))
                device = torch.device('cuda:'+str(i))
                test_tensor = test_tensor.to(device)
                logging.info("Using CUDA device "+str(i))
                print("Using CUDA device "+str(i))
                break
            except BaseException:
                logging.info("Cuda Device "+str(i)+" is busy. Trying other devices")
    else:
        raise EnvironmentError("CUDA NOT AVAILABLE")

    return device


def prepare_dataloader(dataset_class, batch_size, crit, ds_path):
    dataset = dataset_class(root_dir=ds_path, label_file="./Inputs/Labels/subset_mini_imagenet.txt", criterion=crit)        #CHANGED
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - int(len(dataset) * 0.1)

    _, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_ = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_ = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    return train_, val_


def pre_training(opt, config):
    dev = get_cuda_device()
    mm = ResNet(opt.mode, criterion_loss=opt.loss_criterion)

    if opt.loss_criterion == "SupCon":
        model = mm.model
        optimizer = optim.SGD(lr=config["learning_rate"],
                              params=model.parameters(),
                              momentum=config["momentum"],
                              weight_decay=config["L2_decay"],
                              nesterov=config["use_nestrov"])

        criterion = ["SupCon", SupConLoss(temperature=config["temperature"], device=dev), 0]

    elif opt.loss_criterion == "CE":
        model = mm.model
        optimizer = optim.SGD(lr=config["learning_rate"],
                              params=model.parameters(),
                              momentum=config["momentum"],
                              weight_decay=config["L2_decay"],
                              nesterov=config["use_nestrov"])

        criterion = ["CE", F.cross_entropy, 0]
    else:
        raise NameError("Loss not supported")

    train_dl, val_dl = prepare_dataloader(MiniImageNet, opt.batch_size, opt.loss_criterion, source_dataset_path)
    model.double()
    model.to(dev)

    print("STARTING PRETRAINING")
    start_time = time()

    pretrained_model = train(train_dl=train_dl, val_dl=val_dl, criterion=criterion,
                             optimizer=optimizer, device=dev, model=model, config=config, annealing_=opt.mode)
    end_time = time()
    seconds_elapsed = end_time - start_time
    hours, rest = divmod(seconds_elapsed, 3600)
    print("TIME REQUIRED TO PRETRAIN THE MODEL:" + str(hours) + " hrs")

    model_save_path = os.path.join(output_model_path, "pretrained_"+opt.loss_criterion + str(int(start_time)) + ".pt")
    save_model(model_save_path, pretrained_model)


def linear_phase_training(opt, config):
    dev = get_cuda_device()
    mm = ResNet(opt.mode, criterion_loss=opt.loss_criterion, model_name=opt.model_name)
    model = mm.model

    criterion = ["CE", F.cross_entropy, 1]  #FINAL TRAINING WITH CE AS LOSS

    if "SupCon" in opt.model_name:
        model_name = "SupCon"
    else:
        model_name = "CE"
    print("TRAINING STARTED FOR " + model_name)

    optimizer = optim.Adam(params=model.parameters(),
                           lr=config["learning_rate"],
                           amsgrad=opt.use_amsgrad,
                           weight_decay=config["L2_decay"])

    train_dl, val_dl = prepare_dataloader(MiniImageNet, opt.batch_size, "CE", source_dataset_path)
    model.double()
    model.to(dev)

    print("Starting Training")
    start_time = time()

    trained_model = train(train_dl=train_dl, val_dl=val_dl, optimizer=optimizer,
                          device=dev, model=model, criterion=criterion, config=config, annealing_ = opt.mode)
    end_time = time()
    seconds_elapsed = end_time - start_time
    hours, rest = divmod(seconds_elapsed, 3600)

    print("TIME REQUIRED TO TRAIN THE MODEL:" + str(hours) + " hrs")

    model_save_path = os.path.join(output_model_path, "trained_" + model_name + ".pt")
    save_model(model_save_path, trained_model)


def inference(opt):
    dev = get_cuda_device()
    mm = ResNet(opt.mode, criterion_loss=None, model_name=opt.model_name)
    model = mm.model
    test_dl, _ = prepare_dataloader(ImageNetV2, opt.batch_size, opt.loss_criterion, target_dataset_path)
    model.double()
    model.to(dev)

    print("STARTING TESTING")
    start_time = time()

    loss, accuracy = test(val_dl=test_dl, dev=dev, model=model)
    end_time = time()
    seconds_elapsed = end_time - start_time
    _, rest = divmod(seconds_elapsed, 3600)
    minutes, rest = divmod(rest, 60)

    print("ACCURACY = {acc:2.6f}".format(acc=accuracy))
    print("LOSS = {los:2.6f}".format(los=loss))
    print("TIME REQUIRED TO TEST THE MODEL:" + str(minutes) + " mins")


def save_model(save_path, model_to_save):
    torch.save(model_to_save, save_path)
