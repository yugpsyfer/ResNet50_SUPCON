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
            model_with_fc.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=300, bias=True))
            final_model = model_with_fc

        elif self.criterion_loss == "CE":
            model_with_fc.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=100, bias=True),
                                             # nn.ReLU(),
                                             # nn.Linear(in_features=300, out_features=100, bias=True),
                                             nn.Softmax(dim=1))
            final_model = model_with_fc
        else:
            raise ValueError("LOSS NOT SUPPORTED")

        return final_model

    def _make_training_model(self):
        temp_ = self.model_name.split(".")[0]
        temp_ = temp_.split("_")[1]
        if "SupCon" in temp_:
            pretrained_model = self._load_pretrained_model()
            for param in pretrained_model.parameters():
                param.requires_grad = False
            new_linear_1 = nn.Linear(in_features=300, out_features=1024, bias=True)
            new_linear_2 = nn.Linear(in_features=1024, out_features=512, bias=True)
            new_linear_3 = nn.Linear(in_features=512, out_features=256, bias=True)
            new_linear_4 = nn.Linear(in_features=256, out_features=100, bias=True)
            nn.init.xavier_uniform_(new_linear_1.weight)
            nn.init.xavier_uniform_(new_linear_2.weight)
            nn.init.xavier_uniform_(new_linear_3.weight)
            nn.init.xavier_uniform_(new_linear_4.weight)
            pretrained_model.fc.add_module(name="1", module=nn.ReLU())
            pretrained_model.fc.add_module(name="2", module=new_linear_1)
            pretrained_model.fc.add_module(name="3", module=nn.ReLU())
            pretrained_model.fc.add_module(name="4", module=new_linear_2)
            pretrained_model.fc.add_module(name="5", module=nn.ReLU())
            pretrained_model.fc.add_module(name="6", module=new_linear_3)
            pretrained_model.fc.add_module(name="7", module=nn.ReLU())
            pretrained_model.fc.add_module(name="8", module=new_linear_4)
            pretrained_model.fc.add_module(name="9", module=nn.Softmax(dim=1))
        else:
            pretrained_model = self._load_pretrained_model()
            for param in pretrained_model.parameters():
                param.requires_grad = False
            new_linear_1 = nn.Linear(in_features=100, out_features=1000, bias=True)
            new_linear_2 = nn.Linear(in_features=1000, out_features=1000, bias=True)
            new_linear_3 = nn.Linear(in_features=1000, out_features=100, bias=True)
            nn.init.xavier_uniform_(new_linear_1.weight)
            nn.init.xavier_uniform_(new_linear_2.weight)
            nn.init.xavier_uniform_(new_linear_3.weight)
            pretrained_model.fc[1] = nn.ReLU()
            pretrained_model.fc.add_module(name="2", module=new_linear_1)
            pretrained_model.fc.add_module(name="3", module=nn.ReLU())
            pretrained_model.fc.add_module(name="4", module=new_linear_2)
            pretrained_model.fc.add_module(name="5", module=nn.ReLU())
            pretrained_model.fc.add_module(name="6", module=new_linear_3)
            pretrained_model.fc.add_module(name="7", module=nn.Softmax(dim=1))

        final_model = pretrained_model

        return final_model

    def _load_pretrained_model(self):
        output_model_path = './Outputs/Models/'
        load_path = os.path.join(output_model_path, self.model_name)
        model_ = torch.load(load_path)

        return model_

    def _load_trained_model(self):
        output_model_path = './Outputs/Models/'
        load_path = os.path.join(output_model_path, self.model_name)
        model_ = torch.load(load_path)

        return model_
