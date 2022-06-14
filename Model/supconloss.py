"""
Author: Yugansh Singh
Date: 2022, May 23
"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):

    def __init__(self, device, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, features, embeddings, labels):
        embeddings = torch.unsqueeze(embeddings, dim=1)
        features = features.permute(0, 1)

        labels = torch.unsqueeze(labels, dim=1)
        labels = torch.tile(labels, dims=(1,2))
        labels = labels.view(-1, 1)
        label = torch.squeeze(labels)

        label_tile = torch.tile(labels.view(1, -1), dims=(labels.shape[0], 1))
        positive_label_mask = torch.eq(label_tile, label_tile.T)

        negative_label_mask = (~positive_label_mask).to(device=self.device, dtype=torch.int32)
        positive_label_mask = positive_label_mask.to(device=self.device, dtype=torch.int32) - torch.eye(n=labels.shape[0],
                                                                                                        m=labels.shape[0],
                                                                                                        device=self.device)

        # positives = positive_label_mask * torch.broadcast_to(embeddings,  size=(embeddings.shape[0], embeddings.shape[0]))
        # negatives = negative_label_mask * torch.broadcast_to(embeddings,  size=(embeddings.shape[0], embeddings.shape[0]))

        positive_count = torch.sum(positive_label_mask, dim=0)

        F_ = features.expand(features.shape[0], -1, -1)

        all_dot = torch.matmul(embeddings, F_.permute(0, 2, 1))
        all_dot = torch.squeeze(all_dot)

        negatives = all_dot * negative_label_mask
        negatives = negatives / self.temperature
        negatives = torch.exp(negatives)
        negatives = torch.sum(negatives, dim=1)

        positives = all_dot * positive_label_mask
        positives = positives / self.temperature
        positives = torch.exp(positives)

        _log_ = torch.div(positives, negatives)
        _log_ = torch.log(_log_)

        logits = torch.sum(_log_, dim=1)
        logits = torch.div(logits, positive_count)
        logits = logits * (-1)

        loss = torch.sum(logits, dim=0)

        loss = loss / features.shape[0]

        return loss
