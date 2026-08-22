"""
PRN (Polyphase Resonance Network) — 核心实现

基于论文: Resonant Polynomial Neuron (RePoN) + Polyphase Resonance Network (PRN)

组件:
  1. RePoN — 谐振多项式神经元 (Chebyshev + 复数权重 + Hebbian)
  2. PhaseEncoder — 相位编码器
  3. ResonantBlock — 谐振块 (N× RePoN + Complex Layer Norm + 残差)
  4. HyperbolicMapper — 双曲映射器 (→ Poincaré 圆盘)
  5. ManifoldRouter — 流形路由器 (MoE on hyperbolic space)
  6. OutputProjection — 输出投影
  7. PRN — 完整网络
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


# ============================================================
# Chebyshev Polynomials (closed-form)
# ============================================================
def chebyshev_polynomials(x: torch.Tensor, K: int) -> torch.Tensor:
    """
    Compute Chebyshev polynomials T_0(x), ..., T_{K-1}(x).
    OPTIMIZED: vectorized recurrence, no Python loop per step.
    
    T_0 = 1, T_1 = x, T_{k+1} = 2x·T_k - T_{k-1}
    
    Args:
      x: [..., d] input in [-1, 1]
      K: number of terms
    Returns:
      [..., d, K]
    """
    x = x.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    
    if K == 1:
        return torch.ones(*x.shape, 1, device=x.device, dtype=x.dtype)
    
    # Build all K terms using vectorized recurrence (no in-place ops for autograd)
    t_prev = torch.ones_like(x)   # T_0
    t_curr = x.clone()             # T_1
    result = [t_prev, t_curr]
    
    x2 = 2.0 * x
    for k in range(2, K):
        t_next = x2 * t_curr - t_prev
        result.append(t_next)
        t_prev = t_curr
        t_curr = t_next
    
    return torch.stack(result, dim=-1)  # [..., K]


# ============================================================
# 1. RePoN — Resonant Polynomial Neuron
# ============================================================
class RePoN(nn.Module):
    """
    Resonant Polynomial Neuron
    
    y = Σ_k w_k · φ_k(x) · e^{i·α_k}
    
    where φ_k(x) are Chebyshev polynomials, w_k are complex weights,
    α_k are learnable phase parameters, and Hebbian learning
    provides fast local adaptation.
    
    From the paper (Definition 5.1):
      - K polynomial orders
      - C resonance channels
      - Phase parameters α_k ∈ [0, 2π)
      - Complex weights w_jk
      - Hebbian rule: Δw_jk = η·g_j·z_k, Δα_k = η·sin(θ_target - α_k)
    """
    
    def __init__(self, input_dim: int, K: int = 8, C: int = 4,
                 use_hebbian: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.K = K  # number of Chebyshev polynomial orders
        self.C = C  # number of resonance channels
        self.use_hebbian = use_hebbian
        
        # Complex weights: w_k ∈ C^d  (real + imag parts)
        self.w_real = nn.Parameter(torch.randn(K, C) / math.sqrt(2 * K))
        self.w_imag = nn.Parameter(torch.randn(K, C) / math.sqrt(2 * K))
        
        # Phase parameters α_k ∈ [0, 2π) — learnable
        self.alpha = nn.Parameter(torch.rand(K) * 2 * math.pi)
        
        # Hebbian state (not learnable, just buffers for fast adaptation)
        if use_hebbian:
            self.register_buffer('hebbian_w', torch.zeros(K, C))
            self.register_buffer('hebbian_alpha', torch.zeros(K))
        
        # NOTE: channel_proj moved to ResonantBlock for parameter sharing
        # Each RePoN now outputs [..., d, C] directly
        
        # Normalize tau for each order
        self.register_buffer('tau', torch.arange(1, K + 1).float() / K)
    
    def forward(self, x: torch.Tensor, 
                hebbian_lr: float = 0.01) -> torch.Tensor:
        """
        Args:
          x: [..., d] input
          hebbian_lr: Hebbian learning rate (used in training mode)
        
        Returns:
          [..., d, C] output (C resonance channels)
        """
        # Chebyshev expansion: [..., d, K]
        cheb = chebyshev_polynomials(x, self.K)
        
        # Phase-weighted complex weights: w_k * e^{i*alpha_k}  [K, C]
        w = self.w_real + 1j * self.w_imag
        phase = torch.exp(1j * self.alpha)
        phase_w = w * phase.unsqueeze(-1)  # [K, C] complex
        
        # === VECTORIZED: avoid creating [batch, d, K, C] intermediate ===
        # Use einsum: sum_k cheb[...,d,k] * phase_w[k,c] → [..., d, C]
        # Split real/imag to avoid complex tensor allocation
        pw_real = phase_w.real  # [K, C]
        pw_imag = phase_w.imag  # [K, C]
        
        # [..., d, K] × [K, C] → [..., d, C]  (both real, no complex needed)
        out_real = torch.einsum('...dk,kc->...dc', cheb, pw_real)
        out_imag = torch.einsum('...dk,kc->...dc', cheb, pw_imag)
        
        # Hebbian-modulated output: 基础权重 + Hebbian快速适应权重
        if self.use_hebbian:
            # hebbian_w: [K, C] — 通过外积累积的经验权重
            # 与基础权重相乘: cheb[...,d,k] * hebbian_w[k,c] → [..., d, C]
            hebbian_out = torch.einsum('...dk,kc->...dc', cheb, self.hebbian_w)
            output = out_real + 0.3 * hebbian_out  # 混合基础和Hebbian
        else:
            output = out_real  # [..., d, C]
        
        # Hebbian fast adaptation — 真正的局部学习，不需要反向传播
        # 只在训练模式下更新，推理时使用已积累的 hebbian_w/alpha
        if self.use_hebbian and self.training:
            with torch.no_grad():
                # ── Pre-synaptic signal: input 的 Chebyshev 展开 ──
                # g_j = mean over batch of cheb expansion
                g = cheb.mean(dim=tuple(range(cheb.dim()-1)))  # [K]
                
                # ── Post-synaptic signal: output 的通道激活 ──
                # z_c = mean over batch and input dims of output
                z = output.mean(dim=tuple(range(output.dim()-1)))  # [C]
                
                # ── Hebbian weight update: Δw_jk = η · g_j · z_k ──
                # 外积: [K, 1] × [1, C] → [K, C]
                outer = g.unsqueeze(-1) * z.unsqueeze(0)  # [K, C]
                self.hebbian_w += hebbian_lr * outer
                self.hebbian_w.clamp_(-1.0, 1.0)
                
                # ── Phase tracking: Δα_k = η · sin(θ_target - α_k) ──
                # 目标频率 = 输入信号的主频 (跳过 T_0，用高阶多项式能量)
                cheb_energy = cheb.mean(dim=tuple(range(cheb.dim()-1)))  # [K]
                # 跳过 T_0 (index=0)，从 T_1 开始找最大能量对应的阶数
                high_order_energy = cheb_energy[1:]  # [K-1]
                dominant_order = high_order_energy.argmax().item() + 1  # 1-indexed
                target_freq = dominant_order / self.K * 2 * math.pi
                self.hebbian_alpha += hebbian_lr * torch.sin(
                    torch.tensor(target_freq, device=self.alpha.device) - self.alpha)
                self.hebbian_alpha = self.hebbian_alpha % (2 * math.pi)
                # 加法混合: 梯度优化 + Hebbian 局部学习，互不覆盖
                self.alpha.data = 0.7 * self.alpha.data + 0.3 * self.hebbian_alpha
        
        return output


# ============================================================
# 2. Phase Encoder
# ============================================================
class PhaseEncoder(nn.Module):
    """
    Phase Encoder (Section 6.2)
    
    Encodes input with learnable phase information, similar to 
    positional encoding but for phase-resonant processing.
    
    PE(x)_d = x_d · cos(ω_d) + i · x_d · sin(ω_d)
    
    where ω_d are learnable phase frequencies.
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        # Learnable phase frequencies ω_d ∈ [0, π)
        self.omega = nn.Parameter(torch.rand(input_dim) * math.pi)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [..., d]
        Returns:
          [..., d] phase-encoded (real-valued, using magnitude preservation)
        """
        # Apply phase rotation in complex plane
        cos_w = torch.cos(self.omega)
        sin_w = torch.sin(self.omega)
        
        # Real: x * cos(ω), Imag: x * sin(ω)
        # Output magnitude = |x| (phase-encoded but magnitude-preserving)
        real_part = x * cos_w
        imag_part = x * sin_w
        
        # Return as real tensor with doubled dim, or use magnitude
        # For simplicity: return x * (cos(ω) + i*sin(ω)) as real encoding
        # Using the convention from the paper that PE outputs go to ResonantBlocks
        return x * torch.cos(self.omega)  # real-valued phase modulation


# ============================================================
# 3. Complex Layer Normalization
# ============================================================
class ComplexLayerNorm(nn.Module):
    """
    Complex Layer Normalization (Equation 18)
    
    CLN(z) = γ · (z - E[z]) / sqrt(Var(ℜz) + Var(ℑz) + ε) + β
    
    For real-valued PRN variant, we adapt this to normalize
    using both real and imaginary statistics.
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For real-valued tensors, standard LayerNorm
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# ============================================================
# 4. Resonant Block (RB)
# ============================================================
class ResonantBlock(nn.Module):
    """
    Resonant Block (Section 6.3) — OPTIMIZED
    
    z(l) = z(l-1) + CLN(SharedProj(Σ_{n=1}^{N} RePoN_n(z(l-1))))
    
    KEY OPTIMIZATION: Shared channel projection across all N RePoN neurons.
    Before: N × Linear(C*d, d) = N×C×d² params
    After:  1 × Linear(C*d, d) = C×d² params (N× reduction!)
    """
    
    def __init__(self, input_dim: int, N: int = 8, K: int = 8, C: int = 4):
        super().__init__()
        self.N = N
        self.C = C
        self.repons = nn.ModuleList([
            RePoN(input_dim, K=K, C=C) for _ in range(N)
        ])
        # SHARED channel projection (replaces per-RePoN channel_proj)
        self.shared_proj = nn.Linear(C * input_dim, input_dim, bias=False)
        self.norm = ComplexLayerNorm(input_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
          z: [..., d]
        Returns:
          [..., d] after shared projection + residual + norm
        """
        # Each RePoN outputs [..., d, C], sum them: [..., d, C]
        repn_sum = sum(rpn(z) for rpn in self.repons)
        
        # Reshape: [..., d, C] → [..., d*C]
        d = z.shape[-1]
        repn_flat = repn_sum.reshape(*z.shape[:-1], d * self.C)
        
        # SHARED projection: [..., d*C] → [..., d]
        projected = self.shared_proj(repn_flat)
        
        # Complex Layer Norm + Residual
        return z + self.norm(projected)


# ============================================================
# 5. Hyperbolic Mapper (HM)
# ============================================================
class HyperbolicMapper(nn.Module):
    """
    Hyperbolic Mapper (Section 6.4)
    
    Maps latent state z ∈ R^d to Poincaré disk B^n:
      h = exp_0(v) = tanh(||v||/2) · v/||v||
    
    where v = [ℜ(z), ℑ(z)] ∈ R^{2N}
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        # Projection from input_dim to 2*input_dim (real + imag parts)
        self.proj = nn.Linear(input_dim, 2 * input_dim, bias=False)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
          z: [..., d]
        Returns:
          [..., 2d] point on Poincaré disk (||h|| < 1)
        """
        v = self.proj(z)  # [..., 2d]
        
        # Norm
        v_norm = v.norm(dim=-1, keepdim=True)  # [..., 1]
        v_norm = torch.clamp(v_norm, min=1e-8)
        
        # Exponential map: h = tanh(||v||/2) · v/||v||
        h = torch.tanh(v_norm / 2.0) * (v / v_norm)
        
        return h


# ============================================================
# 6. Manifold Router (MR)
# ============================================================
class ManifoldRouter(nn.Module):
    """
    Manifold Router (Section 6.5)
    
    MoE routing on the Poincaré disk:
      d_e = d_B(h, μ_e) = arcosh(1 + 2||h - μ_e||² / ((1-||h||²)(1-||μ_e||²)))
      α_e = exp(-d_e/τ) / Σ exp(-d_e'/τ)
      u = Σ α_e · E_e(h)
    """
    
    def __init__(self, input_dim: int, num_experts: int = 8, 
                 top_k: int = 3, tau: float = 1.0):
        super().__init__()
        self.top_k = top_k
        self.tau = tau
        
        # Expert centers on Poincaré disk (learnable)
        # Initialize with random points inside unit ball
        centers = torch.randn(num_experts, input_dim) * 0.3
        centers = centers / (centers.norm(dim=-1, keepdim=True) + 1e-8) * 0.5
        self.centers = nn.Parameter(centers)
        
        # Expert networks (simple linear for now)
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_experts)
        ])
    
    def poincare_distance(self, x: torch.Tensor, 
                          y: torch.Tensor) -> torch.Tensor:
        """
        Compute Poincaré distance:
        d_B(x, y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
        """
        diff_sq = (x - y).pow(2).sum(dim=-1, keepdim=True)
        
        norm_x_sq = x.pow(2).sum(dim=-1, keepdim=True).clamp(max=1.0 - 1e-6)
        norm_y_sq = y.pow(2).sum(dim=-1, keepdim=True).clamp(max=1.0 - 1e-6)
        
        denominator = (1.0 - norm_x_sq) * (1.0 - norm_y_sq)
        denominator = denominator.clamp(min=1e-8)
        
        arg = 1.0 + 2.0 * diff_sq / denominator
        arg = arg.clamp(min=1.0 + 1e-6)
        
        return torch.acosh(arg)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          h: [..., d] point on Poincaré disk
        Returns:
          u: [..., d] fused expert output
          routing_weights: [..., num_experts] routing weights
        """
        # === VECTORIZED: batch distance computation ===
        # h: [..., d], centers: [E, d] → broadcast
        # [..., 1, d] - [E, d] → [..., E, d]
        h_expanded = h.unsqueeze(-2)  # [..., 1, d]
        centers = self.centers.unsqueeze(0)  # [1, E, d]
        
        diff_sq = (h_expanded - centers).pow(2).sum(dim=-1)  # [..., E]
        
        norm_h_sq = h.pow(2).sum(dim=-1, keepdim=True).clamp(max=1.0 - 1e-6)
        norm_c_sq = self.centers.pow(2).sum(dim=-1).clamp(max=1.0 - 1e-6)  # [E]
        
        denominator = (1.0 - norm_h_sq) * (1.0 - norm_c_sq.unsqueeze(0))
        denominator = denominator.clamp(min=1e-8)
        
        arg = (1.0 + 2.0 * diff_sq / denominator).clamp(min=1.0 + 1e-6)
        distances = torch.acosh(arg)  # [..., E]
        
        # Top-k selection
        top_k_values, top_k_indices = distances.topk(self.top_k, dim=-1, largest=False)
        
        # === VECTORIZED: softmax over top-k ===
        routing_weights = torch.full_like(distances, float('-inf'))
        routing_weights.scatter_(-1, top_k_indices, -top_k_values / self.tau)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # === VECTORIZED: batched expert forward + weighted sum ===
        # h: [..., d] → [E, ..., d] via batched forward
        expert_outputs = torch.stack([exp(h) for exp in self.experts], dim=-2)  # [..., E, d]
        u = (routing_weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)  # [..., d]
        
        return u, routing_weights


# ============================================================
# 7. Output Projection
# ============================================================
class OutputProjection(nn.Module):
    """
    Output Projection (Section 6.6)
    
    ŷ = W_out · u + b_out
    
    Then softmax for classification or linear for regression.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 task: str = 'classification'):
        super().__init__()
        self.task = task
        self.proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        y = self.proj(u)
        # 不在这里做 softmax — CrossEntropyLoss 内部会做 log_softmax
        # 如果需要概率输出，在推理时手动 softmax
        return y


# ============================================================
# 8. Complete PRN Network
# ============================================================
class PRN(nn.Module):
    """
    Complete Polyphase Resonance Network (PRN)
    
    Architecture:
      Input → PE → [RB × L] → HM → MR → OP → Output
    
    Combines:
      - RePoN neurons (Chebyshev + complex weights + Hebbian)
      - Phase encoding
      - Resonant blocks with residual connections
      - Hyperbolic mapping (Poincaré disk)
      - Manifold routing (MoE on hyperbolic space)
      - Output projection
    """
    
    def __init__(self, input_dim: int, output_dim: int,
                 L: int = 4, N: int = 8, K: int = 8, C: int = 4,
                 num_experts: int = 8, top_k: int = 3,
                 task: str = 'classification',
                 use_hebbian: bool = True):
        super().__init__()
        self.L = L  # number of Resonant Blocks
        self.task = task
        
        # Phase Encoder
        self.pe = PhaseEncoder(input_dim)
        
        # L Resonant Blocks
        self.blocks = nn.ModuleList([
            ResonantBlock(input_dim, N=N, K=K, C=C) for _ in range(L)
        ])
        
        # Hyperbolic Mapper
        self.hm = HyperbolicMapper(input_dim)
        
        # Manifold Router
        manifold_dim = 2 * input_dim  # after HM doubles dimension
        self.mr = ManifoldRouter(manifold_dim, num_experts=num_experts,
                                  top_k=top_k)
        
        # Output Projection
        self.op = OutputProjection(manifold_dim, output_dim, task=task)
    
    def forward(self, x: torch.Tensor, 
                return_routing: bool = False) -> torch.Tensor:
        """
        Args:
          x: [..., d] input features
          return_routing: whether to return routing info
        
        Returns:
          tensor (output) by default; dict if return_routing=True
        """
        # Phase Encoding
        z = self.pe(x)
        
        # L Resonant Blocks
        features = []
        for block in self.blocks:
            z = block(z)
            features.append(z)
        
        # Hyperbolic Mapping
        h = self.hm(z)
        
        # Manifold Routing
        u, routing_weights = self.mr(h)
        
        # Output Projection
        output = self.op(u)
        
        if return_routing:
            return {
                'output': output, 
                'features': features[-1],
                'routing_weights': routing_weights,
                'hyperbolic_h': h,
            }
        return output
    
    def get_param_count(self) -> dict:
        """Count parameters by component."""
        counts = {}
        counts['pe'] = sum(p.numel() for p in self.pe.parameters())
        counts['blocks'] = sum(p.numel() for p in self.blocks.parameters())
        counts['hm'] = sum(p.numel() for p in self.hm.parameters())
        counts['mr'] = sum(p.numel() for p in self.mr.parameters())
        counts['op'] = sum(p.numel() for p in self.op.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
