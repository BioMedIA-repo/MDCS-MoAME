import torch
import torch.nn as nn
from mamba.mamba_ssm import IntervalMamba2


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class GenomicMam(nn.Module):
    def __init__(self, in_dim, dropout, act, survival=False, layer=2, delta=5, type="Interval"):
        super(GenomicMam, self).__init__()
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
        self.survival = survival
        if type == "Interval":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(256),
                        IntervalMamba2(
                            d_model=256,
                            d_state=16,
                            d_conv=4,
                            expand=2,

                        ),
                    )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(type))

        self.delta = delta
        self.type = type
        self.apply(initialize_weights)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x.float()  # [B, n, 256]
        if self.type == "Interval":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h)
                h = h + h_
        return h

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers = self.layers.to(device)

        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)
