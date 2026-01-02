import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        B = embeddings.size(0)
        dist = torch.cdist(embeddings, embeddings)

        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.T)
        mask_neg = (labels != labels.T)

        eye = torch.eye(B, device=embeddings.device).bool()
        mask_pos = mask_pos & (~eye)

        hardest_pos = (dist * mask_pos.float()).max(1)[0]
        dist_neg = dist + mask_pos.float() * 1e9
        hardest_neg = dist_neg.min(1)[0]

        return F.relu(hardest_pos - hardest_neg + self.margin).mean()
