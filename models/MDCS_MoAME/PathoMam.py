import torch
import torch.nn as nn

from mamba.mamba_ssm import PatchMam
from mamba.mamba_ssm import RegionMam
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class PathoMam(nn.Module):
    def __init__(self, in_dim, dropout, act, type='patch'):
        super(PathoMam, self).__init__()
        self._fc1 = [nn.Linear(in_dim, 256)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(256)
        self.layers = nn.ModuleList()

        self.layernorm = nn.LayerNorm(256)
        if type == 'patch':
            self.Mamba2_x = PatchMam( d_model=256, d_state=16, d_conv=4,  expand=2)
        elif type == 'region':
            self.Mamba2_x = RegionMam(d_model=256, d_state=16, d_conv=4, expand=2)
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(type))
        self.type = type
        self.apply(initialize_weights)

    def forward(self, x1,x2,x3,x4,x5,coords,add_length):
        if len(x1.shape) == 2:
            x1 = x1.expand(1, -1, -1)
            x2 = x2.expand(1, -1, -1)
            x3 = x3.expand(1, -1, -1)
            x4 = x4.expand(1, -1, -1)
            x5 = x5.expand(1, -1, -1)
        h1 = self.norm(x1.float())  # [B, n, 256]
        h2 = self.norm(x2.float())  # [B, n, 256]
        h3 = self.norm(x3.float())  # [B, n, 256]
        h4 = self.norm(x4.float())  # [B, n, 256]
        h5 = self.norm(x5.float())  # [B, n, 256]

        if self.type == 'patch':
            h0_,h1_,h2_,h3_,h4_,h5_ =self.Mamba2_x(h1, h2, h3, h4, h5, coords, add_length)
            h = h1 + h0_
            return h,h1_,h2_,h3_,h4_,h5_
        if self.type == 'region':
            h = h1 + self.Mamba2_x(h1, h2, h3, h4, h5, coords, add_length)
            return h

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers = self.layers.to(device)

        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)
