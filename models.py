from collections import OrderedDict
from os.path import join
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *


class MILAttentionFCSurvNet(nn.Module):
    def __init__(self, input_dim=512, size_arg="small", dropout=0.25, n_classes=4):
        """
        Multi-instance Learning with Attention for Survival Analysis.

        Args:
            input_dim (int): Dimension size of the input features.
            size_arg (str): Size of the neural network architecture. Choices: "small", "large".
            dropout (float): Dropout rate to prevent overfitting.
            n_classes (int): Number of output classes.
        """
        super(MILAttentionFCSurvNet, self).__init__()

        # Define architecture sizes based on the network size (small or large)
        size_dict = {"small": [input_dim, 256, 256], "large": [1024, 512, 384]}
        size = size_dict[size_arg]

        # Fully connected layers followed by attention network
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = AttnNetGated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)

        # Define the sequential architecture
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout))

        # Classifier layer to output final class predictions
        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        """
        Move model to GPU if available, with multi-GPU support if applicable.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        # Move other components to the device
        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, x):
        """
        Forward pass for the model, including attention mechanism and survival prediction.

        Args:
            x (Tensor): Input features.

        Returns:
            hazards (Tensor): Predicted hazards.
            S (Tensor): Cumulative survival function.
            Y_hat (Tensor): Predicted class labels.
        """
        # Apply attention network and get attention scores and hidden path
        A, h_path = self.attention_net(x)
        A = torch.transpose(A, 1, 0)  # Transpose attention matrix for correct dimensions
        A_raw = A  # Keep raw attention matrix for debugging if necessary
        A = F.softmax(A, dim=1)  # Apply softmax to attention scores
        h_path = torch.mm(A, h_path)  # Weighted sum of features using attention
        h_path = self.rho(h_path).squeeze()

        # Compute class logits
        logits = self.classifier(h_path).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]  # Predicted class with highest score
        hazards = torch.sigmoid(logits)  # Apply sigmoid for hazard prediction
        S = torch.cumprod(1 - hazards, dim=1)  # Cumulative survival function

        return hazards, S, Y_hat, None, None


class LayerAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout=0.25):
        """
        Implements attention mechanism at the layer level.

        Args:
            feature_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension for the attention network.
            dropout (float): Dropout rate.
        """
        super(LayerAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass to compute attention scores and context vector.

        Args:
            x (Tensor): Input features of shape (batch_size, feature_dim).

        Returns:
            context_vector (Tensor): Weighted sum of the input features based on attention.
            attention_scores (Tensor): Attention scores for each input feature.
        """
        attention_scores = self.attention(x)  # (batch_size, 1)
        attention_scores = F.softmax(attention_scores, dim=0)  # Apply softmax
        context_vector = torch.sum(attention_scores * x, dim=0)  # Weighted sum
        context_vector = self.dropout(context_vector)  # Apply dropout
        return context_vector, attention_scores


class ModalityAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout=0.25):
        """
        Implements attention mechanism across modalities.

        Args:
            feature_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension for the attention network.
            dropout (float): Dropout rate.
        """
        super(ModalityAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass to compute attention scores and context vector across modalities.

        Args:
            x (Tensor): Input features of shape (n_modalities, feature_dim).

        Returns:
            context_vector (Tensor): Weighted sum of the input features based on attention.
            attention_scores (Tensor): Attention scores for each modality.
        """
        attention_scores = self.attention(x)  # (n_modalities, 1)
        attention_scores = F.softmax(attention_scores, dim=0)  # Apply softmax
        context_vector = torch.sum(attention_scores * x, dim=0)  # Weighted sum
        context_vector = self.dropout(context_vector)  # Apply dropout
        return context_vector, attention_scores


class MRIFeatureProcessor(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=256, output_dim=256, dropout=0.25):
        """
        Processes MRI features using layer and modality attention.

        Args:
            feature_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension for attention.
            output_dim (int): Output feature dimension.
            dropout (float): Dropout rate.
        """
        super(MRIFeatureProcessor, self).__init__()
        self.layer_attention = LayerAttention(feature_dim, hidden_dim, dropout)
        self.modality_attention = ModalityAttention(feature_dim, hidden_dim, dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, rad_features):
        """
        Forward pass for processing MRI features using attention.

        Args:
            rad_features (list of Tensor): List of features from different modalities.

        Returns:
            final_output (Tensor): Processed output feature vector.
            modality_attention_weights (Tensor): Attention weights for modalities.
            layer_attention_weights_list (list of Tensor): Attention weights for each modality layer.
        """
        modality_context_vectors = []
        modality_attention_weights = []
        layer_attention_weights_list = []

        # Iterate over each modality
        for modality in rad_features:
            layer_context_vector, layer_attention_weights = self.layer_attention(modality)
            modality_context_vectors.append(layer_context_vector)
            layer_attention_weights_list.append(layer_attention_weights)

        modality_context_vectors = torch.stack(modality_context_vectors)  # (n_modalities, feature_dim)
        modality_context_vector, modality_attention_weights = self.modality_attention(modality_context_vectors)

        final_output = self.fc(modality_context_vector)  # (output_dim,)

        return final_output, modality_attention_weights, layer_attention_weights_list


class InstanceAttention(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.25):
        """
        Attention mechanism at the instance level.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension for the attention network.
            dropout (float): Dropout rate.
        """
        super(InstanceAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Forward pass for instance-level attention.

        Args:
            x (Tensor): Input features of shape (batch_size, input_dim).

        Returns:
            context_vector (Tensor): Weighted sum of features based on attention scores.
            attention_scores (Tensor): Attention scores for each instance.
        """
        h = self.fc(x)  # (batch_size, hidden_dim)
        attention_scores = self.attention_net(h)  # (batch_size, 1)
        attention_scores = F.softmax(attention_scores, dim=0)  # Apply softmax
        context_vector = torch.sum(attention_scores * h, dim=0)  # Weighted sum
        return context_vector, attention_scores


class WSIAttentionPooler(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=256, dropout=0.25):
        """
        Attention pooling for whole-slide image (WSI) features.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension for attention mechanism.
            output_dim (int): Output feature dimension.
            dropout (float): Dropout rate.
        """
        super(WSIAttentionPooler, self).__init__()
        self.instance_attention = InstanceAttention(input_dim, hidden_dim, dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def relocate(self):
        """
        Move model to GPU if available, with multi-GPU support if applicable.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.instance_attention = nn.DataParallel(self.instance_attention, device_ids=device_ids).to('cuda:0')
        self.fc = self.fc.to(device)

    def forward(self, x):
        """
        Forward pass to compute instance-level attention and WSI-level features.

        Args:
            x (Tensor): Input features.

        Returns:
            wsi_level_feature (Tensor): Final WSI-level feature vector.
            attention_scores (Tensor): Attention scores for each instance.
        """
        context_vector, attention_scores = self.instance_attention(x)
        wsi_level_feature = self.fc(context_vector)  # (output_dim,)

        return wsi_level_feature, attention_scores


class WSIFeatureProcessor(nn.Module):
    def __init__(self, input_dim=512, size_arg="small", dropout=0.25):
        """
        Process features for whole-slide images using attention pooling.

        Args:
            input_dim (int): Input feature dimension.
            size_arg (str): Size of the neural network architecture. Choices: "small", "large".
            dropout (float): Dropout rate.
        """
        super(WSIFeatureProcessor, self).__init__()
        self.wsi_attention_pooler = WSIAttentionPooler(input_dim, hidden_dim=256, output_dim=256, dropout=dropout)

    def forward(self, x):
        """
        Forward pass to process WSI features.

        Args:
            x (Tensor): Input features.

        Returns:
            x (Tensor): Processed WSI-level feature vector.
            attention_scores (Tensor): Attention scores for each instance.
        """
        x, attention_scores = self.wsi_attention_pooler(x)
        return x, attention_scores


class ClinicalFeatureEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=256, dropout=0.25):
        """
        Encode clinical features for input into the model.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension for the encoding.
            output_dim (int): Output feature dimension.
            dropout (float): Dropout rate.
        """
        super(ClinicalFeatureEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass to encode clinical features.

        Args:
            x (Tensor): Input clinical features.

        Returns:
            (Tensor): Encoded clinical features.
        """
        return self.fc(x)
