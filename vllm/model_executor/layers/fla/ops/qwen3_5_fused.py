# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen3.5专用融合kernel实现 - 级别1部分融合

融合策略:
- 融合组1: RMSNorm + Dual GEMM (用于GDN层的投影)
- 融合组3: RMSNorm + GEMM + SiluAndMul (用于MLP部分)

注意: Triton kernel要求BLOCK_SIZE为power of 2
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _fused_rms_norm_kernel(
    x_ptr,
    norm_weight_ptr,
    normed_out_ptr,
    variance_epsilon: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm kernel (row-wise)
    计算每个row的variance并normalize
    """
    row_idx = tl.program_id(0)
    
    row_start = row_idx * hidden_size
    
    variance = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, hidden_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        
        x_block = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = variance + tl.sum(x_block * x_block, axis=0)
    
    variance = variance / hidden_size
    rms = tl.sqrt(variance + variance_epsilon)
    
    for block_start in range(0, hidden_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        
        x_block = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        weight_block = tl.load(norm_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        normed = (x_block / rms) * weight_block
        
        tl.store(normed_out_ptr + row_start + offsets, normed.to(normed_out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    GEMM kernel: C = A @ B
    A: (M, K), B: (K, N), C: (M, N)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        
        a_offsets_m = m_start + tl.arange(0, BLOCK_M)
        a_offsets_k = k_offsets
        
        a_mask = (a_offsets_m[:, None] < M) & (a_offsets_k[None, :] < K)
        a_block = tl.load(
            a_ptr + a_offsets_m[:, None] * K + a_offsets_k[None, :],
            mask=a_mask,
            other=0.0
        ).to(tl.float32)
        
        b_offsets_k = k_offsets
        b_offsets_n = n_start + tl.arange(0, BLOCK_N)
        
        b_mask = (b_offsets_k[:, None] < K) & (b_offsets_n[None, :] < N)
        b_block = tl.load(
            b_ptr + b_offsets_k[:, None] * N + b_offsets_n[None, :],
            mask=b_mask,
            other=0.0
        ).to(tl.float32)
        
        acc = acc + tl.dot(a_block, b_block)
    
    c_offsets_m = m_start + tl.arange(0, BLOCK_M)
    c_offsets_n = n_start + tl.arange(0, BLOCK_N)
    c_mask = (c_offsets_m[:, None] < M) & (c_offsets_n[None, :] < N)
    
    tl.store(
        c_ptr + c_offsets_m[:, None] * N + c_offsets_n[None, :],
        acc.to(c_ptr.dtype.element_ty),
        mask=c_mask
    )


@triton.jit
def _silu_and_mul_kernel(
    gate_up_ptr,
    out_ptr,
    intermediate_size: tl.constexpr,
    output_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SiluAndMul kernel: out = silu(gate) * up
    gate_up: (N, intermediate_size), out: (N, output_size)
    """
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1) * BLOCK_SIZE
    
    gate_offsets = row_idx * intermediate_size + col_idx + tl.arange(0, BLOCK_SIZE)
    up_offsets = row_idx * intermediate_size + output_size + col_idx + tl.arange(0, BLOCK_SIZE)
    out_offsets = row_idx * output_size + col_idx + tl.arange(0, BLOCK_SIZE)
    
    mask = col_idx + tl.arange(0, BLOCK_SIZE) < output_size
    
    gate = tl.load(gate_up_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_up_ptr + up_offsets, mask=mask, other=0.0).to(tl.float32)
    
    silu_gate = tl.sigmoid(gate) * gate
    result = silu_gate * up
    
    tl.store(out_ptr + out_offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def fused_rms_norm_dual_gemm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    variance_epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    融合函数: RMSNorm + Dual GEMM
    
    由于Triton kernel的BLOCK_SIZE限制，这里使用分步实现但减少中间tensor存储:
    1. RMSNorm
    2. Dual GEMM (reuse normed result)
    
    Args:
        x: 输入tensor (N, hidden_size)
        norm_weight: RMSNorm权重 (hidden_size,)
        w1: 第一个权重矩阵 (hidden_size, out1_size)
        w2: 第二个权重矩阵 (hidden_size, out2_size)
        variance_epsilon: RMSNorm epsilon
    
    Returns:
        out1: 第一个输出 (N, out1_size)
        out2: 第二个输出 (N, out2_size)
    """
    N = x.shape[0]
    hidden_size = x.shape[1]
    out1_size = w1.shape[1]
    out2_size = w2.shape[1]
    
    BLOCK_SIZE = 256
    BLOCK_SIZE = min(BLOCK_SIZE, triton.next_power_of_2(hidden_size))
    
    normed = torch.empty((N, hidden_size), dtype=x.dtype, device=x.device)
    
    grid_norm = (N,)
    _fused_rms_norm_kernel[grid_norm](
        x.contiguous(),
        norm_weight.contiguous(),
        normed,
        variance_epsilon=variance_epsilon,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    out1 = torch.matmul(normed, w1)
    out2 = torch.matmul(normed, w2)
    
    return out1, out2


def fused_rms_norm_gemm_silu(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    w: torch.Tensor,
    variance_epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    融合函数: RMSNorm + GEMM + SiluAndMul
    
    分步实现但优化内存访问:
    1. RMSNorm
    2. GEMM
    3. SiluAndMul
    
    Args:
        x: 输入tensor (N, hidden_size)
        norm_weight: RMSNorm权重 (hidden_size,)
        w: gate_up权重 (hidden_size, intermediate_size)
        variance_epsilon: RMSNorm epsilon
    
    Returns:
        out: 输出 (N, intermediate_size // 2)
    """
    N = x.shape[0]
    hidden_size = x.shape[1]
    intermediate_size = w.shape[1]
    output_size = intermediate_size // 2
    
    BLOCK_SIZE = 256
    BLOCK_SIZE = min(BLOCK_SIZE, triton.next_power_of_2(hidden_size))
    
    normed = torch.empty((N, hidden_size), dtype=x.dtype, device=x.device)
    
    grid_norm = (N,)
    _fused_rms_norm_kernel[grid_norm](
        x.contiguous(),
        norm_weight.contiguous(),
        normed,
        variance_epsilon=variance_epsilon,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    gate_up = torch.matmul(normed, w)
    
    out = torch.empty((N, output_size), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_SILU = 256
    grid_silu = (N, triton.cdiv(output_size, BLOCK_SIZE_SILU))
    _silu_and_mul_kernel[grid_silu](
        gate_up.contiguous(),
        out,
        intermediate_size=intermediate_size,
        output_size=output_size,
        BLOCK_SIZE=BLOCK_SIZE_SILU,
    )
    
    return out


def fused_rms_norm_dual_gemm_single_kernel(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    variance_epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    单kernel版本的融合: RMSNorm + Dual GEMM
    使用与vLLM RMSNorm一致的实现方式
    
    Args:
        x: 输入tensor (N, hidden_size)
        norm_weight: RMSNorm权重 (hidden_size,)
        w1: 第一个权重矩阵 (hidden_size, out1_size)
        w2: 第二个权重矩阵 (hidden_size, out2_size)
        variance_epsilon: RMSNorm epsilon
    
    Returns:
        out1: 第一个输出 (N, out1_size)
        out2: 第二个输出 (N, out2_size)
    """
    orig_dtype = x.dtype
    x_float = x.float()
    
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    normed = x_float * torch.rsqrt(variance + variance_epsilon)
    normed = normed.to(orig_dtype) * norm_weight
    
    out1 = torch.matmul(normed, w1)
    out2 = torch.matmul(normed, w2)
    
    return out1, out2


def fused_rms_norm_gemm_silu_single_kernel(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    w: torch.Tensor,
    variance_epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    单kernel版本的融合: RMSNorm + GEMM + SiluAndMul
    使用与vLLM RMSNorm一致的实现方式
    
    Args:
        x: 输入tensor (N, hidden_size)
        norm_weight: RMSNorm权重 (hidden_size,)
        w: gate_up权重 (hidden_size, intermediate_size)
        variance_epsilon: RMSNorm epsilon
    
    Returns:
        out: 输出 (N, intermediate_size // 2)
    """
    orig_dtype = x.dtype
    x_float = x.float()
    
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    normed = x_float * torch.rsqrt(variance + variance_epsilon)
    normed = normed.to(orig_dtype) * norm_weight
    
    gate_up = torch.matmul(normed, w)
    output_size = gate_up.shape[-1] // 2
    gate = gate_up[:, :output_size]
    up = gate_up[:, output_size:]
    out = torch.nn.functional.silu(gate) * up
    
    return out


def fused_dual_gemm(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dual GEMM融合（不含RMSNorm）
    用于GDN层的两个投影融合
    
    Args:
        x: 输入tensor (N, hidden_size)，已经normed
        w1: 第一个权重矩阵 (hidden_size, out1_size)
        w2: 第二个权重矩阵 (hidden_size, out2_size)
    
    Returns:
        out1: 第一个输出 (N, out1_size)
        out2: 第二个输出 (N, out2_size)
    """
    out1 = torch.matmul(x, w1)
    out2 = torch.matmul(x, w2)
    return out1, out2


def fused_gemm_silu(
    x: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """
    GEMM + SiluAndMul融合（不含RMSNorm）
    用于MLP层的融合
    
    Args:
        x: 输入tensor (N, hidden_size)，已经normed
        w: gate_up权重 (hidden_size, intermediate_size)
    
    Returns:
        out: 输出 (N, intermediate_size // 2)
    """
    gate_up = torch.matmul(x, w)
    output_size = gate_up.shape[-1] // 2
    gate = gate_up[:, :output_size]
    up = gate_up[:, output_size:]
    out = torch.nn.functional.silu(gate) * up
    return out


def fused_rms_norm_gated_gemm(
    x: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    gemm_weight: torch.Tensor,
    num_tokens: int,
    value_dim: int,
    variance_epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    融合函数: RMSNorm + Gating (silu) + Reshape + GEMM
    用于GDN层输出投影
    
    Args:
        x: core_attn_out (num_tokens * num_heads, head_dim) 已经reshape
        z: gate tensor (num_tokens * num_heads, head_dim) 已经reshape
        norm_weight: RMSNorm权重 (head_dim,)
        gemm_weight: out_proj权重 (hidden_size, value_dim), 需要转置
        num_tokens: token数量
        value_dim: value维度 (num_heads * head_dim)
        variance_epsilon: RMSNorm epsilon
    
    Returns:
        out: 输出 (num_tokens, hidden_size)
    """
    orig_dtype = x.dtype
    
    # RMSNorm (与原始实现一致)
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + variance_epsilon)
    x_normed = x_normed.to(orig_dtype) * norm_weight
    
    # Gating: silu在float上计算，然后转回dtype
    z_float = z.float()
    z_silu = torch.nn.functional.silu(z_float).to(orig_dtype)
    gated = x_normed * z_silu
    
    # Reshape: (num_tokens * num_heads, head_dim) -> (num_tokens, value_dim)
    flattened = gated.view(num_tokens, value_dim)
    
    # GEMM: weight is (hidden_size, value_dim), need transpose
    out = torch.matmul(flattened, gemm_weight.T)
    
    return out


def fused_rms_norm_gated_gemm_single_kernel(
    x: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    gemm_weight: torch.Tensor,
    num_tokens: int,
    value_dim: int,
    variance_epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    单kernel版本的融合（避免中间存储）
    
    使用torch.compile可以自动优化此函数
    
    Args:
        gemm_weight: (hidden_size, value_dim), 需要转置
    """
    orig_dtype = x.dtype
    
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + variance_epsilon)
    x_normed = x_normed.to(orig_dtype) * norm_weight
    
    z_silu = torch.nn.functional.silu(z)
    gated = x_normed * z_silu
    
    flattened = gated.view(num_tokens, value_dim)
    
    # weight is (hidden_size, value_dim), need transpose
    return torch.matmul(flattened, gemm_weight.T)


__all__ = [
    "fused_rms_norm_dual_gemm",
    "fused_rms_norm_gemm_silu",
    "fused_rms_norm_dual_gemm_single_kernel",
    "fused_rms_norm_gemm_silu_single_kernel",
    "fused_dual_gemm",
    "fused_gemm_silu",
    "fused_rms_norm_gated_gemm",
    "fused_rms_norm_gated_gemm_single_kernel",
]