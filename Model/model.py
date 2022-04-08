import os
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet:
    def __init__(self, pretrain, criterion_loss, model_name=None, inference=False):
        self.criterion_loss = criterion_loss
        self.model_name = model_name

        if pretrain == 0:
            self.model = self._make_pretraining_model()
        elif pretrain == 1:
            self.model = self._make_training_model()
        else:
            self.model = self._load_trained_model()

    def _make_pretraining_model(self):
        model_with_fc = models.resnet50(pretrained=False)

        if self.criterion_loss == "SupCon":
            model_with_fc.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=300, bias=True), nn.ReLU())
            final_model = model_with_fc

        elif self.criterion_loss == "CE":
            model_with_fc.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=300, bias=True),
                                             nn.ReLU(),
                                             nn.Linear(in_features=300, out_features=100, bias=True),
                                             nn.Softmax(dim=1))
            final_model = model_with_fc
        else:
            raise ValueError("LOSS NOT SUPPORTED")

        return final_model

    def _make_training_model(self):
        if self.model_name.split(".")[0] == "SupCon":
            pretrained_model = self._load_pretrained_model()
            for param in pretrained_model.parameters():
                param.requires_grad = False
            new_linear = nn.Linear(in_features=300, out_features=100, bias=True)
            nn.init.xavier_uniform(new_linear.weight)
            pretrained_model.fc.add_module(name="new_linear", module=new_linear)
            pretrained_model.fc.add_module(nn.Softmax(dim=1))
        else:
            pretrained_model = self._load_pretrained_model()
            for param in pretrained_model.parameters():
                param.requires_grad = False
            new_linear = nn.Linear(in_features=300, out_features=100, bias=True)
            nn.init.xavier_uniform(new_linear.weight)
            pretrained_model.fc[2] = new_linear

        final_model = pretrained_model

        return final_model

    def _load_pretrained_model(self):
        output_model_path = '../Outputs/Models/'
        load_path = os.path.join(output_model_path, self.model_name)
        model_ = torch.load(load_path)

        return model_

    def _load_trained_model(self):
        output_model_path = '../Outputs/Models/'
        load_path = os.path.join(output_model_path, self.model_name)
        model_ = torch.load(load_path)

        return model_
