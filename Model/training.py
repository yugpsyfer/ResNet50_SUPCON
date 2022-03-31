import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def calculate_loss(labels_true, labels_pred):
    loss = F.cross_entropy(labels_true, labels_pred)
    return loss


@torch.no_grad
def validate(val_dl, model, dev):
    net_loss = 0
    net_accuracy = 0
    count = 0

    for batch in val_dl:
        labels, images = batch
        images = images.to_device(dev)
        labels = labels.to_device(dev)

        out = model(images)
        loss = calculate_loss(labels, out)

        pred = torch.argmax(loss, dim=1).cpu()
        pred = pred.numpy().flatten()
        labels = labels.cpu()
        labels = labels.numpy().flatten()

        acc = accuracy_score(y_true=labels, y_pred=pred)

        net_loss += loss.get_item()
        net_accuracy += acc
        count += 1

    return net_loss / count, net_accuracy / count


def train(train_dl, val_dl, epochs, optimizer, model, dev):
    model.train()
    history = []

    for epoch in range(epochs):

        for batch in train_dl:
            optimizer.zero_grad()

            labels, images = batch
            images = images.to_device(dev)
            labels = labels.to_device(dev)

            out = model(images)
            loss = calculate_loss(labels, out)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            l, acc = validate(val_dl, model, dev)
            history.append((l, acc))



