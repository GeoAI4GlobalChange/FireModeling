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
class Fire_module(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim"]

    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.W_i = nn.Linear(input_dim * (n_units + 1), input_dim * n_units)
        self.W_f = nn.Linear(input_dim * (n_units + 1), input_dim * n_units)
        self.W_o = nn.Linear(input_dim * (n_units + 1), input_dim * n_units)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1) * init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1) * init_std)
        self.F_beta = nn.Linear(2 * n_units, 1)
        self.Phi = nn.Linear(2 * n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim

    @torch.jit.script_method
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        c_t = torch.zeros(x.shape[0], self.input_dim * self.n_units).cuda()
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.b_j)
            inp = torch.cat([x[:, t, :], h_tilda_t.view(h_tilda_t.shape[0], -1)], dim=1)
            # eq 2
            i_t = torch.sigmoid(self.W_i(inp))
            f_t = torch.sigmoid(self.W_f(inp))
            o_t = torch.sigmoid(self.W_o(inp))
            # eq 3
            c_t = c_t * f_t + i_t * j_tilda_t.view(j_tilda_t.shape[0], -1)
            # eq 4
            h_tilda_t = (o_t * torch.tanh(c_t)).view(h_tilda_t.shape[0], self.input_dim, self.n_units)
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas * outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas / torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas * mu, dim=1)
        return mean, alphas, betas