import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import math
from itertools import islice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    """
    Negative log-likelihood loss function for survival analysis.
    
    :param hazards: Model output representing hazard rates.
    :param S: Survival function (cumulative survival probabilities).
    :param Y: Ground truth event time bin.
    :param c: Censorship status (1 if event is observed, 0 if censored).
    :param alpha: Weighting parameter for uncensored and censored loss terms.
    :param eps: Small value to prevent log(0) errors.
    
    :return: Computed NLL loss.
    """
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    
    # Calculate survival function if not provided
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # Pad survival function
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    
    # Combine losses with weighting factor alpha
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    return loss.mean()


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    """
    Cross-entropy loss function for survival analysis.
    
    :param hazards: Model output representing hazard rates.
    :param S: Survival function (cumulative survival probabilities).
    :param Y: Ground truth event time bin.
    :param c: Censorship status (1 if event is observed, 0 if censored).
    :param alpha: Weighting parameter for uncensored and censored loss terms.
    :param eps: Small value to prevent log(0) errors.
    
    :return: Computed cross-entropy loss.
    """
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    
    # Calculate survival function if not provided
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # Pad survival function
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y) + eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    
    # Combine losses with weighting factor alpha
    loss = (1 - alpha) * ce_l + alpha * reg
    return loss.mean()


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        return ce_loss(hazards, S, Y, c, alpha or self.alpha)


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        return nll_loss(hazards, S, Y, c, alpha or self.alpha)


class CoxSurvLoss(object):
    def __call__(self, hazards, S, c, **kwargs):
        """
        Cox proportional hazards loss function.
        
        :param hazards: Model output representing hazard rates.
        :param S: Survival times for each patient.
        :param c: Censorship status (1 if event occurred, 0 if censored).
        
        :return: Computed Cox loss.
        """
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        
        # Create risk matrix
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1 - c))
        
        return loss_cox


def l1_reg_all(model, reg_type=None):
    """
    L1 regularization across all model parameters.
    
    :param model: PyTorch model.
    :param reg_type: Regularization type (not used here).
    
    :return: Computed L1 regularization term.
    """
    l1_reg = sum(torch.abs(param).sum() for param in model.parameters())
    return l1_reg


def l1_reg_modules(model, reg_type=None):
    """
    L1 regularization for specific modules in the model (e.g., omic and multimodal layers).
    
    :param model: PyTorch model with different submodules.
    :param reg_type: Regularization type (not used here).
    
    :return: Computed L1 regularization for selected modules.
    """
    l1_reg = 0
    l1_reg += l1_reg_all(model.fc_omic)
    l1_reg += l1_reg_all(model.mm)
    return l1_reg


def device_as(t1, t2):
    """
    Moves tensor t1 to the device of tensor t2.
    
    :param t1: The tensor to move.
    :param t2: The reference tensor.
    
    :return: Tensor t1 on the device of t2.
    """
    return t1.to(t2.device)
