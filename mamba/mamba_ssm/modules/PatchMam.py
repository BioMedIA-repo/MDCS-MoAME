

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    print("causal_conv1d package not found. Using slower PyTorch implementation.")
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin

class TransposeTokenReEmbedding:
    @staticmethod
    def transpose_patch_vec(x, coords, add_length):
        B, N, C = x.shape

        x1 = x[:,:N-add_length,:]
        x2 = x[:,N-add_length:,:]
        x1 = rearrange(x1, "b (n p) l -> b n p l",p=64)

        left_index = 0
        right_index = 0
        result_feature = torch.empty(0).to('cuda')

        for num in range(len(coords[0])):
            right_index = right_index + len(coords[0][num])
            truncated_feature = x1[:,left_index:right_index,:,:]
            left_index = left_index + len(coords[0][num])
            B, N, P, C = truncated_feature.shape
            corrected_truncated_feature  = torch.randn(B, N, P, C).to('cuda')
            for i in range(len(coords[0][num])):
                index = (coords[1][num] == coords[0][num][i]).all(dim=1).nonzero(as_tuple=True)

                corrected_truncated_feature[:,i,:,:] = truncated_feature[:,index,:,:]

            corrected_truncated_feature = rearrange(corrected_truncated_feature, "b n (p1 p2) l -> b n (p2 p1) l",p2=8)
            corrected_truncated_feature = rearrange(corrected_truncated_feature, "b n p l -> b (n p) l")
            if result_feature.numel() == 0:
                result_feature = corrected_truncated_feature
            else:
                result_feature = torch.cat((result_feature,corrected_truncated_feature),dim=1)
        result_feature = torch.cat((result_feature,x2),dim=1)
        return result_feature

    def transpose_patch_left_oblique(x, coords, add_length):
        B, N, C = x.shape
        x1 = x[:,:N-add_length,:]
        x2 = x[:,N-add_length:,:]
        x1 = rearrange(x1, "b (n p) l -> b n p l", p=64)
        left_index = 0
        right_index = 0

        result_feature = torch.empty(0).to('cuda')
        for num in range(len(coords[0])):
            right_index = right_index + len(coords[0][num])
            truncated_feature = x1[:,left_index:right_index,:,:]
            left_index = left_index + len(coords[0][num])
            B, N, P, C = truncated_feature.shape
            corrected_truncated_feature  = torch.randn(B, N, P, C).to('cuda')
            for i in range(len(coords[0][num])):
                index = ( coords[2][num] == coords[0][num][i]).all(dim=1).nonzero(as_tuple=True)[0]
                corrected_truncated_feature[:,i,:,:] = truncated_feature[:,index,:,:]

            index = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
                         13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
                         52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]
            patch_corrected_truncated_feature = torch.randn(B, N, P, C).to('cuda')
            for i in range(N):
                for t in range(len(index)):
                    patch_corrected_truncated_feature[:,i,index[t],:] = corrected_truncated_feature[:,i,t,:]
            patch_corrected_truncated_feature = rearrange(patch_corrected_truncated_feature, "b n p l -> b (n p) l")
            if result_feature.numel() == 0:
                result_feature = patch_corrected_truncated_feature
            else:
                result_feature = torch.cat((result_feature,patch_corrected_truncated_feature),dim=1)
        result_feature = torch.cat((result_feature, x2), dim=1)
        return result_feature

    def transpose_patch_right_oblique(x, coords, add_length):
        B, N, C = x.shape
        x1 = x[:, :N - add_length, :]
        x2 = x[:, N - add_length:, :]
        x1 = rearrange(x1, "b (n p) l -> b n p l", p=64)
        left_index = 0
        right_index = 0

        result_feature = torch.empty(0).to('cuda')
        for num in range(len(coords[0])):
            right_index = right_index + len(coords[0][num])
            truncated_feature = x1[:, left_index:right_index, :, :]
            left_index = left_index + len(coords[0][num])
            B, N, P, C = truncated_feature.shape
            corrected_truncated_feature = torch.randn(B, N, P, C).to('cuda')
            for i in range(len(coords[0][num])):
                index = (coords[3][num] == coords[0][num][i]).all(dim=1).nonzero(as_tuple=True)[0]
                corrected_truncated_feature[:, i, :, :] = truncated_feature[:, index, :, :]

            index = [7, 6, 15, 23, 14, 5, 4, 13, 22, 31, 39, 30, 21, 12, 3, 2, 11, 20, 29, 38, 47, 55, 46, 37, 28, 19,
                     10, 1, 0, 9, 18, 27, 36, 45, 54, 63, 62, 53, 44, 35, 26, 17, 8, 16, 25, 34, 43, 52, 61, 60, 51, 42,
                     33, 24, 32, 41, 50, 59, 58, 49, 40, 48, 57, 56]
            patch_corrected_truncated_feature = torch.randn(B, N, P, C).to('cuda')
            for i in range(N):
                for t in range(len(index)):
                    patch_corrected_truncated_feature[:, i, index[t], :] = corrected_truncated_feature[:, i, t, :]
            patch_corrected_truncated_feature = rearrange(patch_corrected_truncated_feature, "b n p l -> b (n p) l")
            if result_feature.numel() == 0:
                result_feature = patch_corrected_truncated_feature
            else:
                result_feature = torch.cat((result_feature, patch_corrected_truncated_feature), dim=1)
        result_feature = torch.cat((result_feature, x2), dim=1)

        return result_feature

    def transpose_patch_loopback(x, coords, add_length):
        B, N, C = x.shape
        x1 = x[:, :N - add_length, :]
        x2 = x[:, N - add_length:, :]
        x1 = rearrange(x1, "b (n p) l -> b n p l", p=64)
        left_index = 0
        right_index = 0

        result_feature = torch.empty(0).to('cuda')
        for num in range(len(coords[0])):
            right_index = right_index + len(coords[0][num])
            truncated_feature = x1[:, left_index:right_index, :, :]
            left_index = left_index + len(coords[0][num])
            B, N, P, C = truncated_feature.shape
            corrected_truncated_feature = torch.randn(B, N, P, C).to('cuda')
            for i in range(len(coords[0][num])):
                index = (coords[4][num] == coords[0][num][i]).all(dim=1).nonzero(as_tuple=True)[0]
                corrected_truncated_feature[:, i, :, :] = truncated_feature[:, index, :, :]

            index = [7, 6, 15, 23, 14, 5, 4, 13, 22, 31, 39, 30, 21, 12, 3, 2, 11, 20, 29, 38, 47, 55, 46, 37, 28, 19,
                     10, 1, 0, 9, 18, 27, 36, 45, 54, 63, 62, 53, 44, 35, 26, 17, 8, 16, 25, 34, 43, 52, 61, 60, 51, 42,
                     33, 24, 32, 41, 50, 59, 58, 49, 40, 48, 57, 56]
            patch_corrected_truncated_feature = torch.randn(B, N, P, C).to('cuda')
            for i in range(N):
                for t in range(len(index)):
                    patch_corrected_truncated_feature[:, i, index[t], :] = corrected_truncated_feature[:, i, t, :]
            patch_corrected_truncated_feature = rearrange(patch_corrected_truncated_feature, "b n p l -> b (n p) l")
            if result_feature.numel() == 0:
                result_feature = patch_corrected_truncated_feature
            else:
                result_feature = torch.cat((result_feature, patch_corrected_truncated_feature), dim=1)
        result_feature = torch.cat((result_feature, x2), dim=1)

        return result_feature


class PatchMam(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
            self.in_proj2 = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
            self.in_proj3 = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
            self.in_proj4 = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
            self.in_proj5 = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)
            self.in_proj2 = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group,
                                                sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)
            self.in_proj3 = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group,
                                                sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)
            self.in_proj4 = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group,
                                                sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)
            self.in_proj5 = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group,
                                                sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state

        # ----> Forward Conv1d
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

        self.conv1d2 = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d2.weight, -self.conv_init, self.conv_init)

        self.conv1d3 = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d3.weight, -self.conv_init, self.conv_init)

        self.conv1d4 = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d4.weight, -self.conv_init, self.conv_init)

        self.conv1d5 = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d5.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # ----> Forward dt_bias
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

        # Initialize log dt bias
        dt2 = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt2 = torch.clamp(dt2, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt2 = dt2 + torch.log(-torch.expm1(-dt2))
        self.dt2_bias = nn.Parameter(inv_dt2)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt2_bias._no_weight_decay = True

        # Initialize log dt bias
        dt3 = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt3 = torch.clamp(dt3, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt3 = dt3 + torch.log(-torch.expm1(-dt3))
        self.dt3_bias = nn.Parameter(inv_dt3)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt3_bias._no_weight_decay = True

        # Initialize log dt bias
        dt4 = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt4 = torch.clamp(dt4, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt4 = dt4 + torch.log(-torch.expm1(-dt4))
        self.dt4_bias = nn.Parameter(inv_dt4)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt4_bias._no_weight_decay = True

        # Initialize log dt bias
        dt5 = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt5 = torch.clamp(dt5, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt5 = dt5 + torch.log(-torch.expm1(-dt5))
        self.dt5_bias = nn.Parameter(inv_dt5)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt5_bias._no_weight_decay = True

        # ----> Forward A_log
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A2 = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log2 = torch.log(A2).to(dtype=dtype)
        self.A_log2 = nn.Parameter(A_log2)
        self.A_log2._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A3 = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log3 = torch.log(A3).to(dtype=dtype)
        self.A_log3 = nn.Parameter(A_log3)
        self.A_log3._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A4 = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log4 = torch.log(A4).to(dtype=dtype)
        self.A_log4 = nn.Parameter(A_log4)
        self.A_log4._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A5 = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log5 = torch.log(A5).to(dtype=dtype)
        self.A_log5 = nn.Parameter(A_log5)
        self.A_log5._no_weight_decay = True

        # ----> Forward D
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        self.D2 = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D2._no_weight_decay = True

        self.D3 = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D3._no_weight_decay = True

        self.D4 = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D4._no_weight_decay = True

        self.D5 = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D5._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u,u2,u3,u4,u5,coords,add_length, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        zxbcdt2 = self.in_proj(u2)
        zxbcdt3 = self.in_proj(u3)
        zxbcdt4 = self.in_proj(u4)
        zxbcdt5 = self.in_proj(u5)

        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        A2 = -torch.exp(self.A_log2.float())  # (nheads) or (d_inner, d_state)
        A3 = -torch.exp(self.A_log3.float())
        A4 = -torch.exp(self.A_log4.float())
        A5 = -torch.exp(self.A_log5.float())
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # NOTE: Go into this branch
        if self.use_mem_eff_path and inference_params is None:
            out_ = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=None, # to output out_z without the final out_proj
                outproj_bias=None, # to output out_z without the final out_proj
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )

            out2 = mamba_split_conv1d_scan_combined(
                zxbcdt2,
                rearrange(self.conv1d2.weight, "d 1 w -> d w"),
                self.conv1d2.bias,
                self.dt2_bias,
                A2,
                D=rearrange(self.D2, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D2,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=None, # to output out_z without the final out_proj
                outproj_bias=None, # to output out_z without the final out_proj
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )

            out3 = mamba_split_conv1d_scan_combined(
                zxbcdt3,
                rearrange(self.conv1d3.weight, "d 1 w -> d w"),
                self.conv1d3.bias,
                self.dt3_bias,
                A3,
                D=rearrange(self.D3, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D3,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=None, # to output out_z without the final out_proj
                outproj_bias=None, # to output out_z without the final out_proj
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )

            out4 = mamba_split_conv1d_scan_combined(
                zxbcdt4,
                rearrange(self.conv1d4.weight, "d 1 w -> d w"),
                self.conv1d4.bias,
                self.dt4_bias,
                A4,
                D=rearrange(self.D4, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D4,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=None, # to output out_z without the final out_proj
                outproj_bias=None, # to output out_z without the final out_proj
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )

            out5 = mamba_split_conv1d_scan_combined(
                zxbcdt5,
                rearrange(self.conv1d5.weight, "d 1 w -> d w"),
                self.conv1d5.bias,
                self.dt5_bias,
                A5,
                D=rearrange(self.D5, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D5,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=None, # to output out_z without the final out_proj
                outproj_bias=None, # to output out_z without the final out_proj
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )

            out2 = TransposeTokenReEmbedding.transpose_patch_vec(out2, coords, add_length)
            out3 = TransposeTokenReEmbedding.transpose_patch_left_oblique(out3, coords, add_length)
            out4 = TransposeTokenReEmbedding.transpose_patch_right_oblique(out4, coords, add_length)
            out5 = TransposeTokenReEmbedding.transpose_patch_loopback(out5, coords, add_length)

            out = out_ + out2 + out3 + out4 + out5
            out = F.linear(out, self.out_proj.weight, self.out_proj.bias)
            out_ = F.linear(out_, self.out_proj.weight, self.out_proj.bias)
            out2 = F.linear(out2, self.out_proj.weight, self.out_proj.bias)
            out3 = F.linear(out3, self.out_proj.weight, self.out_proj.bias)
            out4 = F.linear(out4, self.out_proj.weight, self.out_proj.bias)
            out5 = F.linear(out5, self.out_proj.weight, self.out_proj.bias)

            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, -(self.dconv - 1):]
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out,out_,out2,out3,out4,out5

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
