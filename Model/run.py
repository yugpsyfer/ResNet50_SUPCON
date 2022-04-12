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
import importlib

source_dataset_path = "./Inputs/mini_image_net_merged/"
target_dataset_path = "./Output/Target/imagenet_v2/"
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
    dataset = dataset_class(root_dir=ds_path, label_file="./Inputs/Labels/wordnet_details.txt", criterion=crit)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - int(len(dataset) * 0.1)

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_ = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_ = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    return train_, val_


def pre_training(opt, config, criterion_options):
    dev = get_cuda_device()
    mm = ResNet(opt.mode, criterion_loss=opt.loss_criterion)

    if opt.loss_criterion == "SupCon":
        model = mm.model
        temperature = criterion_options['SupCon']['temperature']
        base_temperature = temperature
        l_r = criterion_options['SupCon']['learning_rate']
        optimr = importlib.import_module("torch.optim." + criterion_options['SupCon']['optimizer'])
        optimizer = optimr.SGD(lr=l_r, params=model.parameters())
        epochs = criterion_options['SupCon']['epochs']
        criterion = ["SupCon", SupConLoss(temperature=temperature, contrast_mode='all',
                                          base_temperature=base_temperature, device=dev), 0]

    elif opt.loss_criterion == "CE":
        model = mm.model
        epochs = criterion_options['CE']['epochs']
        l_r = criterion_options['CE']['learning_rate']
        optimr = importlib.import_module("torch.optim." + criterion_options['CE']['optimizer'])
        optimizer = optimr.SGD(lr=l_r, params=model.parameters())
        criterion = ["CE", F.cross_entropy, 0]
    else:
        raise NameError("Loss not supported")

    train_dl, val_dl = prepare_dataloader(MiniImageNet, opt.batch_size, opt.loss_criterion, source_dataset_path)
    model.double()
    model.to(dev)
    logging.info("STARTING PRETRAINING")
    print("STARTING PRETRAINING")
    start_time = time()

    pretrained_model = train(train_dl=train_dl, val_dl=val_dl, criterion=criterion, epochs=epochs,
                             optimizer=optimizer, dev=dev, model=model, config=config)
    end_time = time()
    seconds_elapsed = end_time - start_time
    hours, rest = divmod(seconds_elapsed, 3600)
    logging.info("TIME REQUIRED TO PRETRAIN THE MODEL:" + str(hours) + " hrs")

    model_save_path = os.path.join(output_model_path, "pretrained_"+opt.loss_criterion + str(end_time) + ".pt")
    save_model(model_save_path, pretrained_model)


def linear_phase_training(opt, config, criterion_options):
    dev = get_cuda_device()
    mm = ResNet(opt.mode, criterion_loss=opt.loss_criterion, model_name=opt.model_name)
    model = mm.model
    criterion = ["CE", F.cross_entropy, 1]
    if opt.loss_criterion == "SupCon":
        epochs = criterion_options['SupCon']['epochs']
    else:
        epochs = criterion_options['CE']['epochs']

    l_r = 0.0004
    optimizer = optim.Adam(params=model.parameters(), lr=l_r)

    train_dl, val_dl = prepare_dataloader(MiniImageNet, opt.batch_size, "CE", source_dataset_path)
    model.double()
    model.to(dev)

    print("STARTING Training==================")
    logging.info("STARTING TRAINING==================")
    start_time = time()

    trained_model = train(train_dl=train_dl, val_dl=val_dl, epochs=epochs, optimizer=optimizer,
                          dev=dev, model=model, criterion=criterion, config=config)
    end_time = time()
    seconds_elapsed = end_time - start_time
    hours, rest = divmod(seconds_elapsed, 3600)

    logging.info("TIME REQUIRED TO TRAIN THE MODEL:" + str(hours) + " hrs")

    model_save_path = os.path.join(output_model_path, "trained_"+opt.loss_criterion + ".pt")
    save_model(model_save_path, trained_model)


def inference(opt):
    dev = get_cuda_device()
    mm = ResNet(opt.mode, criterion_loss=None, model_name=opt.model_name)
    model = mm.model
    test_dl = prepare_dataloader(ImageNetV2, opt.batch_size, opt.loss_criterion, target_dataset_path)
    model.double()
    model.to(dev)

    print("STARTING TESTING")
    start_time = time()

    loss, accuracy = test(val_dl=test_dl, dev=dev, model=model)
    end_time = time()
    seconds_elapsed = end_time - start_time
    _, rest = divmod(seconds_elapsed, 3600)
    minutes, rest = divmod(rest, 60)

    print("ACCURACY = {accu:2.6f}".format(acc=accuracy))
    print("LOSS = {los:2.6f}".format(los=loss))
    print("TIME REQUIRED TO TEST THE MODEL:" + str(minutes) + " mins")


def save_model(save_path, model_to_save):
    torch.save(model_to_save, save_path)
