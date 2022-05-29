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

    def forward(self, features, embeddings):
        embeddings = torch.unsqueeze(embeddings, dim=1)
        features = features.view(-1, 300)

        mask_embeddings = torch.squeeze(embeddings)
        mask = torch.matmul(mask_embeddings, mask_embeddings.permute(1, 0))
        mask = mask.to(dtype=torch.float32)

        diag = torch.eye(n=mask.shape[0],
                         m=mask.shape[1],
                         dtype=torch.float32,
                         device=self.device)

        mask = torch.sub(mask, diag)     # Removed the diagonal elements
        mask[mask == 1] = 0     # Removed all the positives

        mask[mask != 0] = 1     # Make a mask for only negatives

        positive_mask = torch.ones(size=mask.shape,
                                   dtype=torch.float32,
                                   device=self.device)

        positive_mask = positive_mask - mask - diag

        positive_count = torch.sum(positive_mask, dim=0)

        F_ = features.expand(features.shape[0], -1, -1)

        all_dot = torch.bmm(embeddings, F_.permute(0, 2, 1))
        all_dot = torch.squeeze(all_dot)

        negatives = all_dot * mask
        negatives = negatives / self.temperature
        negatives = torch.exp(negatives)
        negatives = torch.sum(negatives, dim=1)

        positives = all_dot * positive_mask
        positives = positives / self.temperature
        positives = torch.exp(positives)

        _log_ = torch.div(positives, negatives)
        _log_ = torch.log(_log_)

        logits = torch.sum(_log_, dim=1)
        logits = torch.div(logits, positive_count)
        logits = logits * (-1)

        loss = torch.sum(logits, dim=0)

        loss = loss / features.shape[0]  # Average on the entire batch

        return loss
