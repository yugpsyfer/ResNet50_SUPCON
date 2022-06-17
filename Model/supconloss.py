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

        positive_count = torch.sum(positive_label_mask, dim=0)

        all_dot = torch.matmul(embeddings, features.permute(1,0))

        negatives = all_dot
        negatives = negatives / self.temperature
        negatives = torch.exp(negatives) * negative_label_mask
        negatives = torch.sum(negatives, dim=1)

        positives = all_dot
        positives = positives / self.temperature
        positives = torch.exp(positives) * positive_label_mask
        _log_ = torch.div(positives, negatives)
        _log_[_log_ == 0] = 1
        _log_ = torch.log(_log_)

        logits = torch.sum(_log_, dim=1)
        logits = torch.div(logits, positive_count)
        logits = logits * (-1)

        loss = logits.mean()

        # loss = loss / features.shape[0]

        return loss
