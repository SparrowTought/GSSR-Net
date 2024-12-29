import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# cross selective scan ===============================
class SelectiveScanMamba(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        # assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        # all in float
        # if u.stride(-1) != 1:
        #     u = u.contiguous()
        # if delta.stride(-1) != 1:
        #     delta = delta.contiguous()
        # if D is not None and D.stride(-1) != 1:
        #     D = D.contiguous()
        # if B.stride(-1) != 1:
        #     B = B.contiguous()
        # if C.stride(-1) != 1:
        #     C = C.contiguous()
        # if B.dim() == 3:
        #     B = B.unsqueeze(dim=1)
        #     ctx.squeeze_B = True
        # if C.dim() == 3:
        #     C = C.unsqueeze(dim=1)
        #     ctx.squeeze_C = True

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        # dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        # dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanFake(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        x = delta
        out = u
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias = u * 0, delta * 0, A * 0, B * 0, C * 0, C * 0, (
            D * 0 if D else None), (delta_bias * 0 if delta_bias else None)
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


# =============
def antidiagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_scatter(tensor_flat, original_shape):
    # 把斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 创建一个空的张量来存储反向散布的结果
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, H, W]，考虑到需要使用transpose将H和W调换
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_根据expanded_index将元素放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


def antidiagonal_scatter(tensor_flat, original_shape):
    # 把反斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 初始化一个与原始张量形状相同、元素全为0的张量
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, W, H]，因为操作是沿最后一个维度收集的，需要调整形状并交换维度
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_将元素根据索引放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


class CrossScan(torch.autograd.Function):
    # ZSJ 这里是把图像按照特定方向展平的地方，改变扫描方向可以在这里修改
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # xs = x.new_empty((B, 4, C, H * W))
        xs = x.new_empty((B, 8, C, H * W))
        # 添加横向和竖向的扫描
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        # 把横向和竖向的反向部分再反向回来，并和原来的横向和竖向相加
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B, C, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, C, H, W))

        y_res = y_rb + y_da
        # return y.view(B, -1, H, W)
        return y_res


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)

        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B, D, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, D, H, W))

        y_res = y_rb + y_da
        return y_res.view(B, D, -1)
        # return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        # xs = x.new_empty((B, 4, C, L))
        xs = x.new_empty((B, 8, C, L))

        # 横向和竖向扫描
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        # xs = xs.view(B, 4, C, H, W)

        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x.view(B, C, H, W))
        xs[:, 5] = antidiagonal_gather(x.view(B, C, H, W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        # return xs
        return xs.view(B, 8, C, H, W)


# these are for ablations =============
class CrossScan_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys.sum(dim=1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1).contiguous()
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        y = ys.sum(dim=1).view(B, C, H, W)
        return y


class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        y = ys.sum(dim=1).view(B, D, H * W)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.view(B, 1, C, L).repeat(1, 4, 1, 1).contiguous().view(B, 4, C, H, W)
        return xs


# =============
# ZSJ 这里是mamba的具体内容，要增加扫描方向就在这里改
def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        delta_softplus=True,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        # ==============================
        to_dtype=True,  # True: final out to dtype
        force_fp32=False,  # True: input fp32
        # ==============================
        nrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)
    # ZSJ 这里把矩阵拆分成不同方向的序列，并进行扫描
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)
    # ZSJ 这里把处理之后的序列融合起来，并还原回原来的矩阵形式
    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


# =====================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


import math
import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y

class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SimpleDown(nn.Module):
    def __init__(self, in_channels, padding_mode="reflect"):
        super().__init__()
        self.out = nn.Conv2d(
            in_channels,in_channels * 2,kernel_size=2,stride=2,padding=0,bias=False,padding_mode=padding_mode,
        )

    def forward(self, x):
        return self.out(x)

class SimpleUp(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.out = nn.ConvTranspose2d(in_channels,in_channels // 2,kernel_size=2,stride=2,padding=0,bias=False,)

    def forward(self, x):
        return self.out(x)
'''
class GFM(nn.Module):
    def __init__(self, in_channels, feature_num=2, bias=True, padding_mode="reflect"):
        super().__init__()
        self.feature_num = feature_num

        hidden_features = in_channels * feature_num
        self.pwconv = nn.Conv2d(
            hidden_features, hidden_features * 2, 1, 1, 0, bias=bias
        )
        self.dwconv = nn.Conv2d(hidden_features * 2,hidden_features * 2,3,1,1,bias=bias,padding_mode=padding_mode,groups=hidden_features * 2,)
        self.project_out = nn.Conv2d(
            hidden_features, in_channels, kernel_size=1, bias=bias
        )
        self.mlp = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)

    def forward(self, *inp_feats):
        assert len(inp_feats) == self.feature_num
        shortcut = inp_feats[0]
        x = torch.cat(inp_feats, dim=1)
        x = self.pwconv(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return self.mlp(x + shortcut)
'''

class DAF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU()
        )
        self.convs = nn.Sequential(
            nn.Conv2d(2*in_channels, 4*in_channels, 3, padding=1, groups=2*in_channels),
            ChannelAttention(num_feat=4*in_channels),
            nn.Conv2d(4*in_channels, in_channels, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU()
        )
        self.out = nn.Conv2d(in_channels, in_channels, 1)
    def forward(self, cur, pre):
        x = torch.cat([cur, pre], dim=1)
        pre = self.conv1(pre) #c
        x = self.convs(x) * pre
        x = self.conv2(x) + cur
        x = self.out(x)
        return x


class DualSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank


        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=8, merge=True
        )  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=8, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn
        self.alpe1 = nn.Parameter(torch.tensor(0.5))
        self.alpe2 = nn.Parameter(torch.tensor(0.5))
        self.alpe3 = nn.Parameter(torch.tensor(0.5))
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None


    @staticmethod
    def dt_init(dt_rank,d_inner,dt_scale=1.0,dt_init="random",dt_min=0.001,dt_max=0.1,dt_init_floor=1e-4,**factory_kwargs,):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def scan(self, x, rows, cols, mask):
        b, c, h, w = x.shape
        y = torch.zeros((b, c, rows * cols), device=x.device)
        y[:, :, :h * w] = x.view(b, c, -1)
        y = y.view(b, c, rows, cols)
        y = y.permute(0, 1, 3, 2)
        y = y.masked_select(mask).view(b,c,h*w)
        return y

    def multi_selective_scan(
            x: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            nrows=-1,
            delta_softplus=True,
            to_dtype=True,
            multi_scan=None,
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        xs = multi_scan.multi_scan(x)

        L = xs.shape[-1]
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)  # l fixed

        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)


        ys: torch.Tensor = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, L)

        y = multi_scan.merge(ys)

        y = out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)
    def merge(self, x, h, w, rows, cols, mask):
        b,c,_ = x.shape
        y = torch.zeros((b,c,cols,rows),device=x.device)
        y.masked_scatter_(mask, x.view(b,c,h*w))
        y = y.permute(0,1,3,2).contiguous().view(b,c,rows*cols)
        y = y[:,:,:h*w].view(b,c,h,w)
        return y

    def mertxa(self, x, rows, cols, mask, row_indices, col_indices):
        B, C, H, W = x.shape
        L = H * W
        K = 8

        xs = torch.stack([
            torch.where(row_indices == 0,x,torch.flip(x, dims=[3])).view(B, -1, L),
            torch.where(col_indices == 0,torch.transpose(x, dim0=2, dim1=3).contiguous(), torch.flip(torch.transpose(x, dim0=2, dim1=3).contiguous(), dims=[3])).view(B, -1, L)
        ],dim=1).view(B, 2, -1, L)


        #-----------------
        oblique_x, anti_oblique_x = self.scan(x, rows, cols, mask).contiguous().view(B, -1, L), self.scan(torch.flip(x, dims=[-1]), rows, cols, mask).contiguous().view(B, -1, L)
        oblique_x = torch.stack([oblique_x,anti_oblique_x], dim=1).view(B, 2, -1, L)
        xs = torch.stack([xs, oblique_x],dim=1).view(B, 4, -1, L)
        #-----------------

        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (B, 8, C, L)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        return xs,dts,As,Bs,Cs,Ds,dt_projs_bias

    def back(self, out_y, rows, cols, mask, row_indices, col_indices,B, H, W):
        L = H * W
        y1 = out_y[:, 0].view(B, -1, H, W)
        y2 = out_y[:, 1].view(B, -1, W, H)
        y1 = torch.where(row_indices == 0, y1, torch.flip(y1, dims=[3])).view(B, -1, L)
        y2 = torch.transpose(torch.where(col_indices == 0, y2, torch.flip(y2, dims=[3])), dim0=2,
                             dim1=3).contiguous().view(B, -1, L)

        y1_inv = torch.flip(out_y[:, 4], dims=[-1]).view(B, -1, H, W)
        y1_inv = torch.where(row_indices == 0, y1_inv, torch.flip(y1_inv, dims=[3])).view(B, -1, L)
        y2_inv = torch.flip(out_y[:, 5], dims=[-1]).view(B, -1, W, H)
        y2_inv = torch.transpose(torch.where(col_indices == 0, y2_inv, torch.flip(y2_inv, dims=[3])), dim0=2,
                                 dim1=3).contiguous().view(B, -1, L)

        y3 = self.merge(out_y[:, 2], H, W, rows, cols, mask).view(B, -1, L)
        y3_inv = self.merge(torch.flip(out_y[:, 6], dims=[-1]), H, W, rows, cols, mask).view(B, -1, L)

        y4 = self.merge(out_y[:, 3], H, W, rows, cols, mask).view(B, -1, H, W)
        y4 = torch.flip(y4, dims=[-1]).view(B, -1, L)

        y4_inv = self.merge(torch.flip(out_y[:, 7], dims=[-1]), H, W, rows, cols, mask).view(B, -1, H, W)
        y4_inv = torch.flip(y4_inv, dims=[-1]).view(B, -1, L)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y1_inv + y2_inv + y3_inv + y4_inv + y4  # merge the four scaned info (b,e*c,hw)
        y = (
            torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        )  # (b,h,w,e*c)
        y = self.out_norm(y)  # (b,h,w,e*c)
        return y
    def forward_core(self, x, y, istage=0, nums=0):
        B, C, H, W = x.shape
        L = H * W
        K = 8
        i = istage
        n = nums
        rows, cols = math.ceil(H * W / (W + 1)), W + 1
        mask = torch.zeros((B, C, rows * cols), dtype=torch.bool, device=x.device)
        mask[:, :, :H * W] = True
        mask = mask.view(B, C, rows, cols).permute(0, 1, 3, 2) #w,h

        row_indices = torch.arange(x.size(2)).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(-1) % 2
        col_indices = torch.arange(x.size(3)).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(-1) % 2

        xs, dts, As,Bs,Cs,Ds,dt_projs_bias = self.mertxa(x, rows, cols, mask, row_indices, col_indices)
        ys, dtsy, Asy, Bsy, Csy, Dsy, dt_projs_biasy = self.mertxa(y, rows, cols, mask, row_indices, col_indices)

        if n > 1:
            if i == 0:
                
                out_y = self.selective_scan(
                    xs, dts,
                    Asy, Bsy, Cs, Dsy,
                    delta_bias=dt_projs_bias,
                    delta_softplus=True,
                ).view(B, K, -1, L)

                out_y_2 = self.selective_scan(
                    ys, dtsy,
                    As, Bs, Csy, Ds,
                    delta_bias=dt_projs_biasy,
                    delta_softplus=True,
                ).view(B, K, -1, L)
                # assert out_y.dtype == torch.float
            else:
                out_y = self.selective_scan(
                    xs, dts,
                    As, Bs, Cs, Ds,
                    delta_bias=dt_projs_bias,
                    delta_softplus=True,
                ).view(B, K, -1, L)

                out_y_2 = self.selective_scan(
                    ys, dtsy,
                    Asy, Bsy, Csy, Dsy,
                    delta_bias=dt_projs_biasy,
                    delta_softplus=True,
                ).view(B, K, -1, L)
                # assert out_y.dtype == torch.float
        else:
            out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)

            out_y_2 = self.selective_scan(
                ys, dtsy,
                Asy, Bsy, Csy, Dsy,
                delta_bias=dt_projs_biasy,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        assert out_y_2.dtype == torch.float
        out_y = self.back(out_y, rows, cols, mask, row_indices, col_indices, B, H, W)
        out_y_2 = self.back(out_y_2, rows, cols, mask, row_indices, col_indices, B, H, W)
        return out_y, out_y_2

    def forward(self, x1, x2, i, n):
        i = i
        n = n
        with_dconv = (self.d_conv > 1)
        x1 = self.in_proj(x1)
        x2 = self.in_proj(x2)
        x1, z1 = x1.chunk(2, dim=-1)  # (b, h, w, d)
        x2, z2 = x2.chunk(2, dim=-1)  # (b, h, w, d)

        if with_dconv:
            x1 = x1.permute(0, 3, 1, 2).contiguous()
            x1 = self.conv2d(x1)  # (b, d, h, w)
            x2 = x2.permute(0, 3, 1, 2).contiguous()
            x2 = self.conv2d(x2)  # (b, d, h, w)
        x1 = self.act(x1)
        x2 = self.act(x2)
        y1, y2 = self.forward_core(x1, x2, istage=i, nums=n)
        y1 = y1 * z1
        y2 = y2 * z2
        out1 = self.out_proj(y1)
        out2 = self.out_proj(y2)

        return out1, out2



class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        try:
            from ss2d_ablations import SS2DDev
            _OSSM = SS2DDev if forward_type.startswith("dev") else OSSM
        except:
            _OSSM = OSSM

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = _OSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)

    def _forward(self, input: torch.Tensor):
        if self.ssm_branch:
            if self.post_norm:
                x = input + self.drop_path(self.norm(self.op(input)))
            else:
                x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class Decoder_Block(nn.Module):
    """Basic block in decoder."""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel // 2, 'the out_channel is not in_channel//2 in decoder block'
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=in_channel + out_channel, out_channels=out_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  )

    def forward(self, de, en):
        de = self.up(de)
        output = torch.cat([de, en], dim=1)
        output = self.fuse(output)

        return output


class RSM_SS(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",
            use_checkpoint=False,
            **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        if isinstance(norm_layer, str) and norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if isinstance(ssm_act_layer, str) and ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if isinstance(mlp_act_layer, str) and mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        _make_patch_embed = self._make_patch_embed_v2
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)

        _make_downsample = self._make_downsample_v3

        # self.encoder_layers = [nn.ModuleList()] * self.num_layers
        self.encoder_layers = []
        self.decoder_layers = []

        for i_layer in range(self.num_layers):
            # downsample = _make_downsample(
            #     self.dims[i_layer],
            #     self.dims[i_layer + 1],
            #     norm_layer=norm_layer,
            # ) if (i_layer < self.num_layers - 1) else nn.Identity()

            downsample = _make_downsample(
                self.dims[i_layer - 1],
                self.dims[i_layer],
                norm_layer=norm_layer,
            ) if (i_layer != 0) else nn.Identity()  # ZSJ 修改为i_layer != 0，也就是第一层不下采样，和论文的图保持一致，也方便我取出每个尺度处理好的特征

            self.encoder_layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            ))
            if i_layer != 0:
                self.decoder_layers.append(
                    Decoder_Block(in_channel=self.dims[i_layer], out_channel=self.dims[i_layer - 1]))

        self.encoder_block1, self.encoder_block2, self.encoder_block3, self.encoder_block4 = self.encoder_layers
        self.deocder_block1, self.deocder_block2, self.deocder_block3 = self.decoder_layers

        self.upsample_x4 = nn.Sequential(
            nn.Conv2d(self.dims[0], self.dims[0] // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.dims[0] // 2, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out_seg = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(OSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            # ZSJ 把downsample放到前面来，方便我取出encoder中每个尺度处理好的图像，而不是刚刚下采样完的图像
            downsample=downsample,
            blocks=nn.Sequential(*blocks, ),
        ))

    def forward(self, x1: torch.Tensor):
        x1 = self.patch_embed(x1)

        x1_1 = self.encoder_block1(x1)
        x1_2 = self.encoder_block2(x1_1)
        x1_3 = self.encoder_block3(x1_2)
        x1_4 = self.encoder_block4(x1_3)  # b,h,w,c

        x1_1 = rearrange(x1_1, "b h w c -> b c h w").contiguous()
        x1_2 = rearrange(x1_2, "b h w c -> b c h w").contiguous()
        x1_3 = rearrange(x1_3, "b h w c -> b c h w").contiguous()
        x1_4 = rearrange(x1_4, "b h w c -> b c h w").contiguous()

        decode_3 = self.deocder_block3(x1_4, x1_3)
        decode_2 = self.deocder_block2(decode_3, x1_2)
        decode_1 = self.deocder_block1(decode_2, x1_1)

        output = self.upsample_x4(decode_1)
        output = self.conv_out_seg(output)

        return output






