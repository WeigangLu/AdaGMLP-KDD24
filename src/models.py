import copy
import numpy as np
from scipy.stats import entropy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv, GATConv
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AdaGMLP(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            norm_type="none",
            K=1,
    ):
        super(AdaGMLP, self).__init__()
        self.sub_mlps = nn.ModuleList()
        for _ in range(K):
            self.sub_mlps.append(MLP(num_layers,
                                     input_dim,
                                     hidden_dim,
                                     output_dim,
                                     dropout_ratio,
                                     norm_type,
                                     ))

        self.parameters = list(self.sub_mlps.parameters())
        self.K = K

    def forward(self, feats, k):

        return self.sub_mlps[k](feats)


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, norm_type='none'):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
            h_list.append(h)

        return h_list, h


class GCN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            for i in range(num_layers - 2):
                self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
            self.layers.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.dropout(h)
            h_list.append(h)

        return h_list, h


class GAT(nn.Module):
    def __init__(
            self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation, num_heads=4, attn_drop=0.3,
            negative_slope=0.2, residual=False):
        super(GAT, self).__init__()
        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        heads = ([num_heads] * num_layers) + [1]

        self.layers.append(
            GATConv(input_dim, hidden_dim, heads[0], dropout_ratio, attn_drop, negative_slope, False, activation))
        for l in range(1, num_layers - 1):
            self.layers.append(
                GATConv(hidden_dim * heads[l - 1], hidden_dim, heads[l], dropout_ratio, attn_drop, negative_slope,
                        residual, activation))
        self.layers.append(
            GATConv(hidden_dim * heads[-2], output_dim, heads[-1], dropout_ratio, attn_drop, negative_slope, residual,
                    None))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)
            h_list.append(h)

        return h_list, h


class GraphSAGE(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.activation = activation

        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, output_dim, aggregator_type='gcn'))
        else:
            self.layers.append(SAGEConv(input_dim, hidden_dim, aggregator_type='gcn'))
            for i in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='gcn'))
            self.layers.append(SAGEConv(hidden_dim, output_dim, aggregator_type='gcn'))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
            h_list.append(h)

        return h_list, h


class Model(nn.Module):
    def __init__(self, param, model_type=None):
        super(Model, self).__init__()

        if model_type == 'teacher':
            self.model_name = param["teacher"]
        else:
            self.model_name = param["student"]

        if "AdaGMLP" == self.model_name:
            self.encoder = AdaGMLP(
                num_layers=param["num_layers"],
                input_dim=param["feat_dim"],
                hidden_dim=param["hidden_dim_s"],
                output_dim=param["label_dim"],
                dropout_ratio=param["dropout_s"],
                norm_type=param["norm_type"],
                K=param["K"]
            )
        elif "MLP" == self.model_name:
            self.encoder = MLP(
                num_layers=param["num_layers"],
                input_dim=param["feat_dim"],
                hidden_dim=param["hidden_dim"],
                output_dim=param["label_dim"],
                dropout_ratio=param["dropout_s"],
                norm_type=param["norm_type"],
            )
        elif "GCN" == self.model_name:
            self.encoder = GCN(
                num_layers=param["num_layers"],
                input_dim=param["feat_dim"],
                hidden_dim=param["hidden_dim"],
                output_dim=param["label_dim"],
                dropout_ratio=param["dropout_t"],
                activation=F.relu,
            )
        elif "GAT" == self.model_name:
            self.encoder = GAT(
                num_layers=param["num_layers"],
                input_dim=param["feat_dim"],
                hidden_dim=param["hidden_dim"],
                output_dim=param["label_dim"],
                dropout_ratio=param["dropout_t"],
                activation=F.relu,
            )
        elif "SAGE" == self.model_name:
            self.encoder = GraphSAGE(
                num_layers=param["num_layers"],
                input_dim=param["feat_dim"],
                hidden_dim=param["hidden_dim"],
                output_dim=param["label_dim"],
                dropout_ratio=param["dropout_t"],
                activation=F.relu,
            )

    def forward(self, g, feats, k=1):
        if "AdaGMLP" == self.model_name:
            return self.encoder(feats, k)
        elif "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(g, feats)
