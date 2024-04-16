import torch

def lp_loss(pos_pred, neg_pred):
    return -torch.log(pos_pred + 1e-12).mean() - torch.log(1 - neg_pred + 1e-12).mean()