import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import torch.utils.data
import math

class Fire_model(torch.nn.Module):
    def __init__(self, input_dim, index_output_dim,time_lead, n_units, init_std=0.02,teleconnection=False):
        super().__init__()
        self.teleconnection = teleconnection
        if self.teleconnection:
            self.bottleneck_tele = MultiLayerFeedForwardNN(input_dim=4, output_dim=1, num_hidden_layers=0,
                                                           dropout_rate=0.1, hidden_dim=4, activation="relu", )
            self.input_dim = input_dim-3
        else:
            self.input_dim=input_dim
        self.output_dim=index_output_dim
        self.fire_module = Fire_module(self.input_dim, index_output_dim, n_units).cuda()
        self.time_lead=time_lead
        # self.final=nn.Linear(index_output_dim, time_lead)
        # self.final = MultiLayerFeedForwardNN(input_dim=index_output_dim, output_dim=time_lead, num_hidden_layers=0,
        #                                            dropout_rate=0.1, hidden_dim=index_output_dim, activation="relu", )


    def forward(self, x):
        if self.teleconnection:
            tele_fea=self.bottleneck_tele(x[:,:,-4:])
            x=torch.cat((x[:,:,:-4],tele_fea),dim=2)
        fx, alphas, betas = self.fire_module(x)
        # fx = self.final(fx)
        return fx, alphas, betas
class Fire_model(torch.nn.Module):
    def __init__(self, input_dim, index_output_dim,time_lead, n_units, init_std=0.02,teleconnection=False):
        super().__init__()
        self.teleconnection = teleconnection
        if self.teleconnection:
            self.bottleneck_tele = MultiLayerFeedForwardNN(input_dim=4, output_dim=1, num_hidden_layers=0,
                                                           dropout_rate=0.1, hidden_dim=4, activation="relu", )
            self.input_dim = input_dim-3
        else:
            self.input_dim=input_dim
        self.output_dim=index_output_dim
        self.fire_module = Fire_module(self.input_dim, index_output_dim, n_units).cuda()
        self.time_lead=time_lead
        # self.final=nn.Linear(index_output_dim, time_lead)
        # self.final = MultiLayerFeedForwardNN(input_dim=index_output_dim, output_dim=time_lead, num_hidden_layers=0,
        #                                            dropout_rate=0.1, hidden_dim=index_output_dim, activation="relu", )


    def forward(self, x):
        if self.teleconnection:
            tele_fea=self.bottleneck_tele(x[:,:,-4:])
            x=torch.cat((x[:,:,:-4],tele_fea),dim=2)
        fx, alphas, betas = self.fire_module(x)
        # fx = self.final(fx)
        return fx, alphas, betas