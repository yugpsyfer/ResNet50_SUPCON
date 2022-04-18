import torch
from sklearn.metrics import accuracy_score
import wandb
from time import time


def calculate_loss(criterion, labels_true, out, embeddings_=None):
    loss_name = criterion[0]
    loss_func = criterion[1]

    if loss_name == "CE":
        loss = loss_func(out, labels_true)
    elif loss_name == "SupCon":
        bsz = labels_true.shape[0]
        f1, f2 = torch.split(out, [bsz, bsz], dim=0)
        out = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = loss_func(features=out, embeddings=embeddings_, labels=labels_true)

    return loss


@torch.no_grad()
def validate(val_dl, model, dev, criterion):
    net_loss = 0
    net_accuracy = 0
    count = 0

    for batch in val_dl:

        if criterion[0] == "SupCon":
            images, labels, embeddings = batch
            embeddings = torch.cat([embeddings[0], embeddings[1]], dim=0)
            embeddings = embeddings.type(torch.DoubleTensor)
            images = torch.cat([images[0], images[1]], dim=0)
            images = images.type(torch.DoubleTensor)
            embeddings = embeddings.to(dev)

        elif criterion[0] == "CE":
            embeddings = None
            images, labels = batch
            images = images.type(torch.DoubleTensor)

        images = images.to(dev)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(dev)

        out = model(images)
        loss = calculate_loss(criterion, labels, out, embeddings_=embeddings)

        # if criterion[2] == 1:
        predicted = torch.argmax(out, dim=1).cpu()
        predicted = predicted.numpy().flatten()
        labels = labels.cpu()
        labels = labels.numpy().flatten()
        acc = accuracy_score(y_true=labels, y_pred=predicted)
        net_accuracy += acc

        loss = torch.nan_to_num(loss)
        net_loss += loss.item()

        count += 1

    return net_loss / count, net_accuracy / count


def train(train_dl, val_dl, epochs, optimizer, model, dev, criterion, config):

    wandb.watch(model, log_freq=10)
    wandb.run.name = config['criterion'] + str(int(time()))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1,
    #                                                        verbose=False)
    model.train()
    history = dict()
    history['train'] = []
    for epoch in range(epochs):

        for batch in train_dl:
            optimizer.zero_grad()

            if criterion[0] == "SupCon":
                images, labels, embeddings = batch
                embeddings = torch.cat([embeddings[0], embeddings[1]], dim=0)
                embeddings = embeddings.type(torch.DoubleTensor)
                images = torch.cat([images[0], images[1]], dim=0)
                images = images.type(torch.DoubleTensor)
                embeddings = embeddings.to(dev)

            elif criterion[0] == "CE":
                embeddings = None
                images, labels = batch
                images = images.type(torch.DoubleTensor)
            else:
                raise ValueError("ERROR LOSS FUNCTION")

            images = images.to(dev)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(dev)

            out = model(images)
            loss = calculate_loss(criterion, labels, out, embeddings)
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            l_val, acc_val = validate(val_dl, model, dev, criterion)
            l_train, acc_train = validate(train_dl, model, dev, criterion)
            # if criterion[] == 0:
            #     wandb.log({"Average_Loss": l_val*100})
            # else:
            wandb.log({"Validation Loss": l_val,
                       "Validation Accuracy": acc_val*100,
                       "Training Loss": l_train,
                       "Training Accuracy": acc_train*100})
        # scheduler.step()
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
        loss = calculate_loss("CE", labels, out, embeddings_=None)

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
