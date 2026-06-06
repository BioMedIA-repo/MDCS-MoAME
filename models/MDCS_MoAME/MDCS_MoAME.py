import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .util import initialize_weights
from .util import SNN_Block
from .util import MultiheadAttention

from .GenomicMam import GenomicMam
from .PathoMam import PathoMam
from .Cross_Mamba import CrossMamba

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        feat_token =x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

class PatchMam(nn.Module):
    def __init__(self, feature_dim=512):
        super(PatchMam, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.PathoMam = PathoMam(in_dim=256,dropout=0.5,act="relu",type='patch')
        self.norm = nn.LayerNorm(feature_dim)
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, features, coords):
        # ---->pad
        H = features[1].shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h1 = torch.cat([features[1], features[1][:, :add_length, :]], dim=1)
        h2 = torch.cat([features[3], features[1][:, :add_length, :]], dim=1)
        h3 = torch.cat([features[5], features[1][:, :add_length, :]], dim=1)
        h4 = torch.cat([features[7], features[1][:, :add_length, :]], dim=1)
        h5 = torch.cat([features[9], features[1][:, :add_length, :]], dim=1)

        h,h1,h2,h3,h4,h5 = self.PathoMam(h1,h2,h3,h4,h5,coords,add_length)
        h = self.pos_layer(h, _H, _W)

        h = self.norm(h)
        A = self.attention(h)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1)
        cls = torch.bmm(A, h)
        cls = cls.squeeze(0)
        return cls

class RegionMam(nn.Module):
    def __init__(self, feature_dim=512):
        super(RegionMam, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.PathoMam = PathoMam(in_dim=256,dropout=0.5,act="relu",type='region')
        self.norm = nn.LayerNorm(feature_dim)
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, features, coords):
        # ---->pad
        H = features[2].shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        #
        h1 = torch.cat([features[2], features[2][:, :add_length, :]], dim=1)
        h2 = torch.cat([features[4], features[2][:, :add_length, :]], dim=1)
        h3 = torch.cat([features[6], features[2][:, :add_length, :]], dim=1)
        h4 = torch.cat([features[8], features[2][:, :add_length, :]], dim=1)
        h5 = torch.cat([features[10], features[2][:, :add_length, :]], dim=1)

        h = self.PathoMam(h1,h2,h3,h4,h5,coords,add_length)
        h = self.pos_layer(h, _H, _W)

        h = self.norm(h)
        A = self.attention(h)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1)
        cls = torch.bmm(A, h)
        cls = cls.squeeze(0)
        return cls

class MDCS_GeneMam(nn.Module):
    def __init__(self, feature_dim=512):
        super(MDCS_GeneMam, self).__init__()
        # Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.GenomicMam = GenomicMam(in_dim=256, dropout=0.5, act="relu", survival=True, layer=2, delta=5,
                                  type="Interval")
        self.norm = nn.LayerNorm(feature_dim)
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, features):
        h = features
        h = self.GenomicMam(h)

        # ---->cls_token
        h = self.norm(h)
        A = self.attention(h)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1)
        cls = torch.bmm(A, h)
        cls = cls.squeeze(0)
        return cls

class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class CroAttFusion(nn.Module):
    def __init__(self, dim = 256):
        super().__init__()
        self.cross_attention = MultiheadAttention(embed_dim=dim, num_heads=1)
    def forward(self, x1, x2):
        x, _ = self.cross_attention(x1.unsqueeze(0).transpose(1, 0), x2.unsqueeze(0).transpose(1, 0), x2.unsqueeze(0).transpose(1, 0))
        return x.transpose(1, 0).squeeze(0)

class CroMamFusion(nn.Module):
    def __init__(self, dim = 256):
        super().__init__()
        self.Cross_Mamba = CrossMamba(dim=dim, mamba_type="m3", headdim=64)
    def forward(self, x1, x2):
        x = self.Cross_Mamba(x1, x2)
        return x

class StackedMamFusion(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=256):
        super().__init__()
        self.norm = norm_layer(dim)
        self.Cross_Mamba1 = CrossMamba(dim=dim, mamba_type="m3", headdim=64)
        self.Cross_Mamba2 = CrossMamba(dim=dim, mamba_type="m3", headdim=64)
        self.bottleneck = torch.rand((1,dim)).cuda()

    def forward(self, x1, x2):
        black = self.Cross_Mamba1(self.bottleneck, x2)
        x = self.Cross_Mamba2(x1, self.norm(black))
        return x

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

class Gating_network(nn.Module):
    def __init__(self, branch_num, norm_layer=RMSNorm, dim=256):
        super(Gating_network, self).__init__()
        self.bnum = branch_num
        self.fc1 = nn.Sequential(
            *[
                nn.Linear(dim, dim),
                norm_layer(dim),
                nn.GELU(),
            ]
        )
        self.fc2 = nn.Sequential(
            *[
                nn.Linear(dim, dim),
                norm_layer(dim),
                nn.GELU(),
            ]
        )
        self.clsfer = nn.Linear(dim, branch_num)

    def forward(self, x1, x2, temp=1.0, hard=False):
        x1, x2 = self.fc1(x1), self.fc2(x2)
        x = x1 + x2
        x = self.clsfer(x)
        logits = DiffSoftmax(x, tau=temp, hard=hard, dim=1)
        return logits

class MoAME(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=256):
        super().__init__()
        self.CroMamFusion = CroMamFusion(dim)
        self.CroAttFusion = CroAttFusion(dim)
        self.StackedMamFusion = StackedMamFusion(norm_layer, dim)
        self.routing_network = Gating_network(3, dim=dim)
        self.routing_dict = {
            0: self.CroAttFusion,
            1: self.CroMamFusion,
            2: self.StackedMamFusion,
        }

    def forward(self, x1, x2, hard=False):
        logits = self.routing_network(x1, x2, hard) # [0.1, 0.3, 0.4, 0.2]
        if hard:
            corresponding_net_id = torch.argmax(logits, dim=1).item()
            x = self.routing_dict[corresponding_net_id](x1, x2)
        else:
            x = torch.zeros_like(x1)
            for branch_id, branch in self.routing_dict.items():
                x += branch(x1, x2)
        return x

class MDCS_MoAME(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, model_size="small"):
        super(MDCS_MoAME, self).__init__()

        self.omic_sizes = omic_sizes
        self.n_classes = n_classes


        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }

        # Pathomics Embedding
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_patch_fc = nn.Sequential(*fc)
        fc1 = []
        for idx in range(len(hidden) - 1):
            fc1.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc1.append(nn.ReLU())
            fc1.append(nn.Dropout(0.25))
        self.pathomics_region_fc = nn.Sequential(*fc1)

        # Genomic Embedding
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        # Enhancement
        self.PatchMam = PatchMam(feature_dim=hidden[-1])
        self.RegionMam = RegionMam(feature_dim=hidden[-1])

        # Enhancement
        self.MDCS_GeneMam = MDCS_GeneMam(feature_dim=hidden[-1])

        ### MoAME
        self.MoAME_genom1 = MoAME(dim=hidden[-1])
        self.MoAME_patho1 = MoAME(dim=hidden[-1])
        self.MoAME_genom2 = MoAME(dim=hidden[-1])
        self.MoAME_patho2 = MoAME(dim=hidden[-1])

        # Classification Layer
        self.mm = nn.Sequential(
            *[nn.Linear(hidden[-1] * 3, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
        )
        self.classifier = nn.Linear( hidden[-1], self.n_classes)
        self.apply(initialize_weights)


    def forward(self, **kwargs):
        # Meta genomics and pathomics features
        x_path = [kwargs["x_path%d" % i] for i in range(1, 11)] # [53437,1024]
        coords = [kwargs["coord%d" % i] for i in range(1, 6)]
        x_path.insert(0,1)
        x_omic = [kwargs["x_omic%d" % i] for i in range(1, 7)]
        index_num = kwargs["index_num"][0]
        separated_coords = []
        for num in range(len(coords)): # 5
            coords_result = []
            left_index = 0
            right_index = 0
            for value in index_num:
                right_index = right_index + value
                coords_result.append(coords[num][left_index:right_index])
                left_index = left_index + value
            separated_coords.append(coords_result)

        # Genomics Embedding
        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0)  # [1, 6, 256]
        # Pathomics Embedding
        pathomics_features = [None] * len(x_path)
        for i in range(1, 11, 2):
            pathomics_features[i] = self.pathomics_patch_fc(x_path[i]).unsqueeze(0)
        for i in range(0, 11, 2):
            if i == 0:
                continue
            pathomics_features[i] = self.pathomics_region_fc(x_path[i]).unsqueeze(0)

        # Pathomics Enhancement
        I_r = self.RegionMam(pathomics_features,separated_coords)
        I_p = self.PatchMam(pathomics_features, separated_coords)
        # Genomics Enhancement
        G_g = self.MDCS_GeneMam(genomics_features)

        # Expert-driven Cross-modal Interaction
        I_r_g = self.MoAME_patho1(I_r, G_g, hard=True)
        I_p_g = self.MoAME_patho2(I_p, G_g, hard=True)
        G_g_r = self.MoAME_genom1(G_g, I_r_g, hard=True)
        G_g_p = self.MoAME_genom2(G_g, I_p_g, hard=True)


        # Feature Alignment and Prediction
        fusion = self.mm(
            torch.concat(
                (
                    (I_p + I_p_g) / 2,
                    (I_r + I_r_g) / 2,
                    (G_g * 2 + G_g_p + G_g_r) / 4,
                ),
                dim=1,
            )
        )  # take cls token to make prediction

        # predict
        logits = self.classifier(fusion)  # [1, n_classes]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S, I_p, I_p_g, I_r, I_r_g, G_g, G_g_p, G_g_r