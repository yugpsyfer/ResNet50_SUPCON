"""TO DO
ADD get item for tensor loss in SUPCON METRIC

"""
import torch
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import wandb
from time import time
import torch.optim.lr_scheduler as Scheduler


def calculate_loss(criterion, labels_true, out, embeddings_):
    loss_name = criterion[0]
    loss_func = criterion[1]

    if loss_name == "CE":
        loss = loss_func(out, labels_true)
    elif loss_name == "SupCon":
        out = F.normalize(out, dim=-1)
        loss = loss_func(features=out, embeddings=embeddings_,
                         labels=labels_true)

    return loss


def metric(y_true, y_pred, embeddings, criterion):

    if criterion[0] == "CE":

        predicted = torch.argmax(y_pred, dim=1).cpu()
        predicted = predicted.numpy().flatten()
        labels = y_true.cpu()
        labels = labels.numpy().flatten()

        return accuracy_score(y_true=labels, y_pred=predicted)

    elif criterion[0] == "SupCon":
        y_pred = F.normalize(y_pred, dim=-1)
        _m = F.cosine_similarity(embeddings, y_pred)
        _m = torch.sum(_m, dim=0)/_m.shape[0]
        return _m


@torch.no_grad()
def validate(val_dl, model, device, criterion):
    net_loss = 0
    net_metric_ = 0
    count = 0

    for batch in val_dl:

        if criterion[0] == "SupCon":
            images, embeddings, labels = batch
            embeddings = torch.cat([embeddings[0], embeddings[1]], dim=0)
            embeddings = embeddings.type(torch.DoubleTensor)
            images = torch.cat([images[0], images[1]], dim=0)
            images = images.type(torch.DoubleTensor)
            embeddings = embeddings.to(device)
            labels = labels.to(device)

        elif criterion[0] == "CE":
            embeddings = None
            images, labels = batch
            images = images.type(torch.DoubleTensor)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)

        images = images.to(device)

        out = model(images)
        loss = calculate_loss(criterion, labels, out, embeddings_=embeddings)
        metric_ = metric(labels, out, embeddings, criterion)

        net_metric_ += metric_

        # loss = torch.nan_to_num(loss)
        net_loss += loss.item()

        count += 1

    return net_loss / count, net_metric_ / count


def train(train_dl, val_dl, optimizer, model, device, criterion, config, annealing_):

    wandb.watch(model, log_freq=10)
    wandb.run.name = config['criterion'] + str(int(time()))

    if annealing_ == 0:
        scheduler = Scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                T_max=config["epochs"],
                                                eta_min=config["min_lr"])

    model.train()

    for epoch in range(config["epochs"]):

        for batch in train_dl:
            optimizer.zero_grad()

            if criterion[0] == "SupCon":
                images, embeddings, labels = batch
                embeddings = torch.cat([embeddings[0], embeddings[1]], dim=0)
                embeddings = embeddings.type(torch.DoubleTensor)
                images = torch.cat([images[0], images[1]], dim=0)
                images = images.type(torch.DoubleTensor)
                embeddings = embeddings.to(device)
                labels = labels.to(device)

            elif criterion[0] == "CE":
                embeddings = None
                images, labels = batch
                images = images.type(torch.DoubleTensor)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
            else:
                raise ValueError("ERROR LOSS FUNCTION")

            images = images.to(device)

            out = model(images)
            loss = calculate_loss(criterion, labels, out, embeddings)
            loss.backward()
            optimizer.step()

        l_val, metric_val = validate(val_dl, model, device, criterion)
        l_train, metric_train = validate(train_dl, model, device, criterion)
        # print("TRAIN LOSS = ", l_train)
        # print("TRAIN Cosine sim = ", metric_train)
        wandb.log({"Validation Loss": l_val,
                   "Validation " + config["metric"]: metric_val,
                   "Training Loss": l_train,
                   "Training " + config["metric"]: metric_train})

        if annealing_ == 0:
            scheduler.step()

    return model


@torch.no_grad()
def test(val_dl, model, dev):
    net_loss = 0
    net_accuracy = 0
    count = 0

    for batch in val_dl:
        images, labels = batch

        images = images.to(dev)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(dev)

        out = model(images)
        loss = calculate_loss(criterion=["CE", F.cross_entropy, 2], labels_true=labels,out=out, embeddings_=None)

        pred = torch.argmax(out, dim=1).cpu()
        pred = pred.numpy().flatten()
        labels = labels.cpu()
        labels = labels.numpy().flatten()

        acc = accuracy_score(y_true=labels, y_pred=pred)
        loss = torch.nan_to_num(loss)
        net_loss += loss.item()
        net_accuracy += acc
        count += 1

    return net_loss / count, net_accuracy / count
