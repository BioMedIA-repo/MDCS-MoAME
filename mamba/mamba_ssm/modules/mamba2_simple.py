

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class Mamba2Simple(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        # headdim=128,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,

        bimamba_type="v2",
        if_devide_out=False,
        init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out
        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.bimamba_type == 'v3' or self.bimamba_type == "swap_c" or self.bimamba_type == "flip" or self.bimamba_type == "fuse" or self.bimamba_type == "m3" or self.bimamba_type == "m3_c":
            self.in_proj_b = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
            self.conv1d_b = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            if self.conv_init is not None:
                nn.init.uniform_(self.conv1d_b.weight, -self.conv_init, self.conv_init)
            # self.conv1d.weight._no_weight_decay = True
            if self.learnable_init_states:
                self.init_states_b = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
                self.init_states_b._no_weight_decay = True
            dt_b = torch.exp(
                torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt_b = torch.clamp(dt_b, min=dt_init_floor)
            inv_dt_b = dt_b + torch.log(-torch.expm1(-dt_b))
            self.dt_bias_b = nn.Parameter(inv_dt_b)
            self.dt_bias_b._no_weight_decay = True
            # A parameter
            A_b = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
            A_log_b = torch.log(A_b).to(dtype=dtype)
            self.A_log_b = nn.Parameter(A_log_b)
            self.A_log_b._no_weight_decay = True
            # D "skip" parameter
            self.D_b = nn.Parameter(torch.ones(self.nheads, device=device))
            self.D_b._no_weight_decay = True
            assert RMSNormGated is not None
            self.norm_b = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
            self.out_proj_b = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        if  self.bimamba_type == "fuse" or self.bimamba_type == "m3_c":
            self.in_proj_c = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
            self.conv1d_c = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            if self.conv_init is not None:
                nn.init.uniform_(self.conv1d_c.weight, -self.conv_init, self.conv_init)
            # self.conv1d.weight._no_weight_decay = True
            if self.learnable_init_states:
                self.init_states_c = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
                self.init_states_c._no_weight_decay = True
            dt_c = torch.exp(
                torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt_c = torch.clamp(dt_c, min=dt_init_floor)
            inv_dt_c = dt_c + torch.log(-torch.expm1(-dt_c))
            self.dt_bias_c = nn.Parameter(inv_dt_c)
            self.dt_bias_c._no_weight_decay = True
            # A parameter
            A_c = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
            A_log_c = torch.log(A_c).to(dtype=dtype)
            self.A_log_c = nn.Parameter(A_log_c)
            self.A_log_c._no_weight_decay = True
            # D "skip" parameter
            self.D_c = nn.Parameter(torch.ones(self.nheads, device=device))
            self.D_c._no_weight_decay = True
            assert RMSNormGated is not None
            self.norm_c = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
            self.out_proj_c = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
            self.out_c = nn.Linear(2 * self.d_model, self.d_model, bias=bias, **factory_kwargs)
            self.out_cc = nn.Linear(3 * self.d_model, self.d_model, bias=bias, **factory_kwargs)

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        self.swap_c1 = nn.Linear(self.d_model, self.d_model, bias=bias, **factory_kwargs)
        self.swap_c2 = nn.Linear(self.d_model, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, seq_idx=None, extra_emb=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape
        if self.bimamba_type == "flip":
            # print(u.shape)[1,1,256]
            u = torch.cat((u,extra_emb),dim = 2)
            # print(u.shape)[1,1,512]
            extra_emb = torch.flip(u, dims = [-1])



        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.bimamba_type == "v3" or self.bimamba_type == "swap_c" or self.bimamba_type == "flip" or self.bimamba_type == "fuse" or self.bimamba_type == "m3" or self.bimamba_type == "m3_c":
            zxbcdt_b = self.in_proj_b(extra_emb)  # (B, L, d_in_proj)
            A_b = -torch.exp(self.A_log_b)  # (nheads) or (d_inner, d_state)
            initial_states_b = repeat(self.init_states_b, "... -> b ...", b=batch) if self.learnable_init_states else None
            dt_limit_kwargs_b = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.bimamba_type == "fuse"  or self.bimamba_type == "m3_c":
            u_fuse = F.linear(torch.cat((u, extra_emb), dim=-1), self.out_c.weight, self.out_c.bias)
            # print(u_fuse.shape)
            zxbcdt_c = self.in_proj_c(u_fuse)  # (B, L, d_in_proj)
            A_c = -torch.exp(self.A_log_c)  # (nheads) or (d_inner, d_state)
            initial_states_c = repeat(self.init_states_c, "... -> b ...", b=batch) if self.learnable_init_states else None
            dt_limit_kwargs_c = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.use_mem_eff_path:
            if self.bimamba_type == "v3" or self.bimamba_type == "flip":
                # Fully fused path
                out = mamba_split_conv1d_scan_combined(
                    zxbcdt,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.eps,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=False,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
                # Fully fused path
                out_b = mamba_split_conv1d_scan_combined(
                    zxbcdt_b,
                    rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    self.conv1d_b.bias,
                    self.dt_bias_b,
                    A_b,
                    D=self.D_b,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm_b.weight,
                    rmsnorm_eps=self.norm_b.eps,
                    outproj_weight=self.out_proj_b.weight,
                    outproj_bias=self.out_proj_b.bias,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=False,
                    initial_states=initial_states_b,
                    **dt_limit_kwargs_b,
                )
                if not self.if_devide_out:
                    out = F.linear(torch.cat((out, out_b), dim=-1), self.out_proj.weight, self.out_proj.bias)
                else:
                    out = F.linear(rearrange(out + out_b, "b d l -> b l d") / 2, self.out_proj.weight,
                                   self.out_proj.bias)

            elif self.bimamba_type == "swap_c":

                c_left = 2 * self.d_inner + self.ngroups * self.d_state
                c_right = 2 * self.d_inner + 2 * self.ngroups * self.d_state

                zxbcdt[:, :, c_left:c_right], zxbcdt_b[:, :, c_left:c_right] = zxbcdt_b[:, :, c_left:c_right].clone(), zxbcdt[:, :,c_left:c_right].clone()
                # Fully fused path
                out = mamba_split_conv1d_scan_combined(
                    zxbcdt,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.eps,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=False,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
                # Fully fused path
                out_b = mamba_split_conv1d_scan_combined(
                    zxbcdt_b,
                    rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    self.conv1d_b.bias,
                    self.dt_bias_b,
                    A_b,
                    D=self.D_b,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm_b.weight,
                    rmsnorm_eps=self.norm_b.eps,
                    outproj_weight=self.out_proj_b.weight,
                    outproj_bias=self.out_proj_b.bias,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=False,
                    initial_states=initial_states_b,
                    **dt_limit_kwargs_b,
                )

                if not self.if_devide_out:


                    out = F.linear(out, self.swap_c1.weight, self.swap_c1.bias)
                    out_b = F.linear(out_b, self.swap_c2.weight, self.swap_c2.bias)
                    out = torch.cat((out, out_b), dim=-1)
                else:

                    out = F.linear(rearrange(out + out_b, "b d l -> b l d") / 2, self.out_proj.weight,
                                   self.out_proj.bias)

            elif self.bimamba_type == "fuse":

                c_left = 2 * self.d_inner + self.ngroups * self.d_state
                c_right = 2 * self.d_inner + 2 * self.ngroups * self.d_state
                zxbcdt[:, :, c_left:c_right] = zxbcdt_c[:, :, c_left:c_right]
                zxbcdt_b[:, :, c_left:c_right] = zxbcdt_c[:, :,c_left:c_right]
                # Fully fused path
                out = mamba_split_conv1d_scan_combined(
                    zxbcdt,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.eps,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=False,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
                # Fully fused path
                out_b = mamba_split_conv1d_scan_combined(
                    zxbcdt_b,
                    rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    self.conv1d_b.bias,
                    self.dt_bias_b,
                    A_b,
                    D=self.D_b,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm_b.weight,
                    rmsnorm_eps=self.norm_b.eps,
                    outproj_weight=self.out_proj_b.weight,
                    outproj_bias=self.out_proj_b.bias,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=False,
                    initial_states=initial_states_b,
                    **dt_limit_kwargs_b,
                )
                out_c = mamba_split_conv1d_scan_combined(
                    zxbcdt_c,
                    rearrange(self.conv1d_c.weight, "d 1 w -> d w"),
                    self.conv1d_c.bias,
                    self.dt_bias_c,
                    A_c,
                    D=self.D_c,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm_c.weight,
                    rmsnorm_eps=self.norm_c.eps,
                    outproj_weight=self.out_proj_c.weight,
                    outproj_bias=self.out_proj_c.bias,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=False,
                    initial_states=initial_states_c,
                    **dt_limit_kwargs_c,
                )
                # print(out.shape)
                # print(out_b.shape)
                if not self.if_devide_out:
                    # out = F.linear(rearrange(out + out_b, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                    out = F.linear(torch.cat((out, out_b, out_c), dim=-1), self.out_cc.weight, self.out_cc.bias)
                else:
                    out = F.linear(rearrange(out + out_b + out_c, "b d l -> b l d") / 3, self.out_cc.weight,
                                   self.out_proj_c.bias)

            elif self.bimamba_type == "m3":
                c_left = 2 * self.d_inner
                c_right = 2 * self.d_inner + self.ngroups * self.d_state
                zxbcdt[:, :, c_left:c_right] = zxbcdt_b[:, :, c_left:c_right]
                # Fully fused path
                A = (A + A_b) / 2
                out = mamba_split_conv1d_scan_combined(
                    zxbcdt,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.eps,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=False,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )

            elif self.bimamba_type == "m3_c":
                c_left = 2 * self.d_inner
                c_right = 2 * self.d_inner + self.ngroups * self.d_state
                zxbcdt[:, :, c_left:c_right] = zxbcdt_c[:, :, c_left:c_right]
                zxbcdt_b[:, :, c_left:c_right] = zxbcdt_c[:, :, c_left:c_right]
                A = (A + A_c) / 2
                A_b = (A_b + A_c) / 2
                # Fully fused path
                out = mamba_split_conv1d_scan_combined(
                    zxbcdt,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.eps,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=False,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
                # Fully fused path
                out_b = mamba_split_conv1d_scan_combined(
                    zxbcdt_b,
                    rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    self.conv1d_b.bias,
                    self.dt_bias_b,
                    A_b,
                    D=self.D_b,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm_b.weight,
                    rmsnorm_eps=self.norm_b.eps,
                    outproj_weight=self.out_proj_b.weight,
                    outproj_bias=self.out_proj_b.bias,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=False,
                    initial_states=initial_states_b,
                    **dt_limit_kwargs_b,
                )

                if not self.if_devide_out:
                    out = F.linear(torch.cat((out, out_b), dim=-1), self.out_proj.weight, self.out_proj.bias)
                else:
                    out = F.linear(rearrange(out + out_b, "b d l -> b l d") / 2, self.out_proj.weight,
                                   self.out_proj.bias)

            else:
                # Fully fused path
                out = mamba_split_conv1d_scan_combined(
                    zxbcdt,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.eps,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=False,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
        else:
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Multiply "gate" branch and apply extra normalization layer
            y = self.norm(y, z)
            out = self.out_proj(y)
        return out
