"""
Qwen3.5融合kernel单元测试
验证融合kernel的正确性（使用single_kernel版本）
"""
import torch
from vllm.model_executor.layers.layernorm import rms_norm
from vllm.model_executor.layers.fla.ops.qwen3_5_fused import (
    fused_rms_norm_dual_gemm_single_kernel,
    fused_rms_norm_gemm_silu_single_kernel,
)


def test_fused_rms_norm_dual_gemm():
    """测试融合组1: RMSNorm + Dual GEMM"""
    
    torch.manual_seed(42)
    
    N = 128
    hidden_size = 2560
    out1_size = 5120
    out2_size = 512
    eps = 1e-6
    
    x = torch.randn(N, hidden_size, device='cuda', dtype=torch.bfloat16)
    norm_weight = torch.randn(hidden_size, device='cuda', dtype=torch.bfloat16)
    w1 = torch.randn(hidden_size, out1_size, device='cuda', dtype=torch.bfloat16)
    w2 = torch.randn(hidden_size, out2_size, device='cuda', dtype=torch.bfloat16)
    
    normed = rms_norm(x, norm_weight, eps)
    
    out1_eager = torch.matmul(normed, w1)
    out2_eager = torch.matmul(normed, w2)
    
    out1_fused, out2_fused = fused_rms_norm_dual_gemm_single_kernel(
        x.contiguous(),
        norm_weight.contiguous(),
        w1.contiguous(),
        w2.contiguous(),
        variance_epsilon=eps,
    )
    
    assert out1_fused.shape == out1_eager.shape
    assert out2_fused.shape == out2_eager.shape
    
    atol = 1.0  
    rtol = 0.01
    
    out1_match = torch.allclose(out1_fused, out1_eager, atol=atol, rtol=rtol)
    out2_match = torch.allclose(out2_fused, out2_eager, atol=atol, rtol=rtol)
    
    if not out1_match:
        diff1 = torch.abs(out1_fused - out1_eager)
        max_diff1 = diff1.max()
        mean_diff1 = diff1.mean()
        print(f"out1 max diff: {max_diff1}, mean diff: {mean_diff1}")
    
    if not out2_match:
        diff2 = torch.abs(out2_fused - out2_eager)
        max_diff2 = diff2.max()
        mean_diff2 = diff2.mean()
        print(f"out2 max diff: {max_diff2}, mean diff: {mean_diff2}")
    
    assert out1_match, f"out1 mismatch"
    assert out2_match, f"out2 mismatch"
    
    print("✓ fused_rms_norm_dual_gemm test passed!")


def test_fused_rms_norm_gemm_silu():
    """测试融合组3: RMSNorm + GEMM + SiluAndMul"""
    
    torch.manual_seed(42)
    
    N = 128
    hidden_size = 2560
    intermediate_size = 4096
    output_size = intermediate_size // 2
    eps = 1e-6
    
    x = torch.randn(N, hidden_size, device='cuda', dtype=torch.bfloat16)
    norm_weight = torch.randn(hidden_size, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(hidden_size, intermediate_size, device='cuda', dtype=torch.bfloat16)
    
    normed = rms_norm(x, norm_weight, eps)
    
    gate_up_eager = torch.matmul(normed, w)
    
    gate_eager = gate_up_eager[:, :output_size]
    up_eager = gate_up_eager[:, output_size:]
    out_eager = torch.nn.functional.silu(gate_eager) * up_eager
    
    out_fused = fused_rms_norm_gemm_silu_single_kernel(
        x.contiguous(),
        norm_weight.contiguous(),
        w.contiguous(),
        variance_epsilon=eps,
    )
    
    assert out_fused.shape == out_eager.shape
    
    atol = 100.0  
    rtol = 0.1
    
    out_match = torch.allclose(out_fused, out_eager, atol=atol, rtol=rtol)
    
    if not out_match:
        diff = torch.abs(out_fused - out_eager)
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"out max diff: {max_diff}, mean diff: {mean_diff}")
    
    assert out_match, f"out mismatch"
    
    print("✓ fused_rms_norm_gemm_silu test passed!")


def test_small_batch():
    """测试小batch场景"""
    
    torch.manual_seed(42)
    
    N = 1
    hidden_size = 2560
    out1_size = 5120
    out2_size = 512
    eps = 1e-6
    
    x = torch.randn(N, hidden_size, device='cuda', dtype=torch.bfloat16)
    norm_weight = torch.randn(hidden_size, device='cuda', dtype=torch.bfloat16)
    w1 = torch.randn(hidden_size, out1_size, device='cuda', dtype=torch.bfloat16)
    w2 = torch.randn(hidden_size, out2_size, device='cuda', dtype=torch.bfloat16)
    
    out1_fused, out2_fused = fused_rms_norm_dual_gemm_single_kernel(
        x.contiguous(),
        norm_weight.contiguous(),
        w1.contiguous(),
        w2.contiguous(),
        variance_epsilon=eps,
    )
    
    normed = rms_norm(x, norm_weight, eps)
    out1_eager = torch.matmul(normed, w1)
    out2_eager = torch.matmul(normed, w2)
    
    assert torch.allclose(out1_fused, out1_eager, atol=1e-2, rtol=1e-2)
    assert torch.allclose(out2_fused, out2_eager, atol=1e-2, rtol=1e-2)
    
    print("✓ Small batch test passed!")


def test_performance_comparison():
    """性能对比测试"""
    
    torch.manual_seed(42)
    
    N = 128
    hidden_size = 2560
    out1_size = 5120
    out2_size = 512
    intermediate_size = 4096
    eps = 1e-6
    
    x = torch.randn(N, hidden_size, device='cuda', dtype=torch.bfloat16)
    norm_weight = torch.randn(hidden_size, device='cuda', dtype=torch.bfloat16)
    w1 = torch.randn(hidden_size, out1_size, device='cuda', dtype=torch.bfloat16)
    w2 = torch.randn(hidden_size, out2_size, device='cuda', dtype=torch.bfloat16)
    w_mlp = torch.randn(hidden_size, intermediate_size, device='cuda', dtype=torch.bfloat16)
    
    import time
    
    torch.cuda.synchronize()
    
    iterations = 100
    
    start = time.time()
    for _ in range(iterations):
        normed = rms_norm(x, norm_weight, eps)
        out1 = torch.matmul(normed, w1)
        out2 = torch.matmul(normed, w2)
    torch.cuda.synchronize()
    eager_time_dual = time.time() - start
    
    start = time.time()
    for _ in range(iterations):
        out1_fused, out2_fused = fused_rms_norm_dual_gemm_single_kernel(
            x, norm_weight, w1, w2, variance_epsilon=eps
        )
    torch.cuda.synchronize()
    fused_time_dual = time.time() - start
    
    start = time.time()
    for _ in range(iterations):
        normed = rms_norm(x, norm_weight, eps)
        gate_up = torch.matmul(normed, w_mlp)
        gate = gate_up[:, :intermediate_size//2]
        up = gate_up[:, intermediate_size//2:]
        out = torch.nn.functional.silu(gate) * up
    torch.cuda.synchronize()
    eager_time_mlp = time.time() - start
    
    start = time.time()
    for _ in range(iterations):
        out_fused = fused_rms_norm_gemm_silu_single_kernel(
            x, norm_weight, w_mlp, variance_epsilon=eps
        )
    torch.cuda.synchronize()
    fused_time_mlp = time.time() - start
    
    print("\n性能对比结果:")
    print(f"  融合组1 (RMSNorm + Dual GEMM):")
    print(f"    Eager时间: {eager_time_dual:.4f}s ({iterations}次)")
    print(f"    Fused时间: {fused_time_dual:.4f}s ({iterations}次)")
    speedup_dual = (eager_time_dual - fused_time_dual) / eager_time_dual * 100
    print(f"    性能变化: {speedup_dual:+.2f}%")
    
    print(f"  融合组3 (RMSNorm + GEMM + SiluAndMul):")
    print(f"    Eager时间: {eager_time_mlp:.4f}s ({iterations}次)")
    print(f"    Fused时间: {fused_time_mlp:.4f}s ({iterations}次)")
    speedup_mlp = (eager_time_mlp - fused_time_mlp) / eager_time_mlp * 100
    print(f"    性能变化: {speedup_mlp:+.2f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3.5融合kernel单元测试")
    print("=" * 60)
    
    test_fused_rms_norm_dual_gemm()
    test_fused_rms_norm_gemm_silu()
    test_small_batch()
    test_performance_comparison()
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)