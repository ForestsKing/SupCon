import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        :param features: hidden vector of shape [batch_size, view_num, embed_dim]
        :return: A loss scalar.
        """
        device = features.device
        batch_size = features.shape[0]
        view_num = features.shape[1]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)

        features = F.normalize(features, dim=-1)
        distance = torch.matmul(features, features.T)
        exp_distance = torch.exp(distance / self.temperature)

        pos_mask = torch.eye(batch_size).repeat(view_num, view_num).float().to(device)
        pos_mask = pos_mask - torch.eye(batch_size * view_num).float().to(device)
        all_mask = torch.ones_like(pos_mask).float().to(device)
        all_mask = all_mask - torch.eye(batch_size * view_num).float().to(device)

        exp_distance_sum = (exp_distance * all_mask).sum(-1, keepdim=True)
        loss_all = torch.log(exp_distance / exp_distance_sum)
        loss = (loss_all * pos_mask).sum(-1) / pos_mask.sum(-1)
        loss = -loss.mean()

        return loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        :param features: hidden vector of shape [batch_size, view_num, embed_dim]
        :param labels: ground truth of shape [batch_size]
        :return: A loss scalar.
        """
        device = features.device
        batch_size = features.shape[0]
        view_num = features.shape[1]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        labels = labels.contiguous().view(-1, 1)

        features = F.normalize(features, dim=-1)
        distance = torch.matmul(features, features.T)
        exp_distance = torch.exp(distance / self.temperature)

        pos_mask = torch.eq(labels, labels.T).repeat(view_num, view_num).float().to(device)
        pos_mask = pos_mask - torch.eye(batch_size * view_num).float().to(device)
        all_mask = torch.ones_like(pos_mask).float().to(device)
        all_mask = all_mask - torch.eye(batch_size * view_num).float().to(device)

        exp_distance_sum = (exp_distance * all_mask).sum(-1, keepdim=True)
        loss_all = torch.log(exp_distance / exp_distance_sum)
        loss = (loss_all * pos_mask).sum(-1) / pos_mask.sum(-1)
        loss = -loss.mean()

        return loss
