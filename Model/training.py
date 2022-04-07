import torch
from sklearn.metrics import accuracy_score
import logging


def calculate_loss(criterion, labels_true, out, embeddings_):
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
        images, labels = batch
        images = images.to(dev)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(dev)

        out = model(images)
        loss = calculate_loss(criterion, labels, out)

        pred = torch.argmax(out, dim=1).cpu()
        pred = pred.numpy().flatten()
        labels = labels.cpu()
        labels = labels.numpy().flatten()

        acc = accuracy_score(y_true=labels, y_pred=pred)

        net_loss += loss.item()
        net_accuracy += acc
        count += 1

    return net_loss / count, net_accuracy / count


def train(train_dl, val_dl, epochs, optimizer, model, dev, criterion):
    model.train()
    history = dict()
    history['train'] = []

    for epoch in range(epochs):

        for batch in train_dl:
            optimizer.zero_grad()

            if criterion[0] == "SupCon":
                images, labels, embeddings = batch
                embeddings = embeddings.to(dev)
            elif criterion[1] == "CE":
                images, labels = batch

            images = images.to(dev)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(dev)

            out = model(images)
            loss = calculate_loss(criterion, labels, out, embeddings)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            # l, acc = validate(train_dl, model, dev, criterion)
            l_train, acc_train = validate(train_dl, model, dev, criterion)
            logging.info("####################################################################################")
            logging.info("EPOCH: {epch}".format(epch=epoch))
            logging.info("------------------------------------------------------------------------------------")
            logging.info("TRAIN LOSS: {error:2.6f}\nTRAIN ACCURACY: {accu:2.6f}".format(error=l_train, accu=acc_train))
            logging.info("####################################################################################")
            # history['val'].append((l, acc))
            history['train'].append((l_train, acc_train))

    return model, history



