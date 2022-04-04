import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def calculate_loss(criterion, labels_true, out):
    loss_name = criterion[0]
    loss_func = criterion[1]

    if loss_name == "CE":
        loss = loss_func(labels_true, out)
    elif loss_name == "SupCon":
        bsz = labels_true.shape[0]
        out = F.normalize(out, dim=1)
        f1, f2 = torch.split(out, [bsz, bsz], dim=0)
        out = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = loss_func(features=out, labels=labels_true)

    return loss


@torch.no_grad()
def validate(val_dl, model, dev, criterion):
    net_loss = 0
    net_accuracy = 0
    count = 0

    for batch in val_dl:
        labels, images = batch
        images = images.to(dev)
        labels = labels.to(dev)

        out = model(images)
        loss = calculate_loss(criterion, labels, out)

        pred = torch.argmax(out, dim=1).cpu()
        pred = pred.numpy().flatten()
        labels = labels.cpu()
        labels = labels.numpy().flatten()

        acc = accuracy_score(y_true=labels, y_pred=pred)

        net_loss += loss.get_item()
        net_accuracy += acc
        count += 1

    return net_loss / count, net_accuracy / count


def train(train_dl, val_dl, epochs, optimizer, model, dev, criterion):
    model.train()
    history = dict()
    history['train'] = []
    history['val'] = []

    for epoch in range(epochs):

        for batch in train_dl:
            optimizer.zero_grad()

            labels, images = batch
            images = images.to(dev)
            labels = labels.to(dev)

            out = model(images)
            loss = calculate_loss(criterion, labels, out)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            l, acc = validate(val_dl, model, dev, criterion)
            history['val'].append((l, acc))

            history['train'].append(loss)

    return model, history



