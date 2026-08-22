"""
QuantNN — 量化交易模型

架构:
  1. Regime Router (L1): 识别市场状态 → 牛/震荡/熊
  2. Expert Pool: 8个异构专家 (CNN/LSTM/MLP/PRN/GNN/AE/XGB/Factor)
  3. Expert Fusion: 加权融合选中专家的输出
  4. Action Head: 输出交易信号 (LONG/SHORT/FLAT/HOLD)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


# ============================================================
# Expert Base
# ============================================================
class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ============================================================
# 1. CNN Expert — K线形态识别
# ============================================================
class CNNExpert(Expert):
    def __init__(self, input_dim, hidden_dim=64, output_dim=4):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, output_dim)
    
    def forward(self, x):
        # x: [B, D] → reshape to [B, 1, D] for conv1d
        h = x.unsqueeze(1)
        h = self.conv(h).squeeze(-1)
        return self.fc(h)


# ============================================================
# 2. LSTM Expert — 时序动量
# ============================================================
class LSTMExpert(Expert):
    def __init__(self, input_dim, hidden_dim=64, output_dim=4):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: [B, D] → treat as single timestep
        h, _ = self.lstm(x.unsqueeze(1))
        return self.fc(h[:, -1])


# ============================================================
# 3. MLP Expert — 多因子融合
# ============================================================
class MLPExpert(Expert):
    def __init__(self, input_dim, hidden_dim=64, output_dim=4):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. PRN Expert — 谐振多项式 (集成已有实现)
# ============================================================
class PRNExpert(Expert):
    """简化版 PRN 专家 — 直接内联 Chebyshev + 相位 + 简单路由"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=4, K=8, C=4):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.K = K
        self.C = C
        
        # Chebyshev weights + phase
        self.w_real = nn.Parameter(torch.randn(K, C) / math.sqrt(2*K))
        self.w_imag = nn.Parameter(torch.randn(K, C) / math.sqrt(2*K))
        self.alpha = nn.Parameter(torch.rand(K) * 2 * math.pi)
        
        # Shared projection
        self.proj = nn.Linear(C * input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Chebyshev expansion
        xc = x.clamp(-0.9, 0.9)  # tighter clamp for stability
        T_list = [torch.ones_like(xc), xc.clone()]
        x2 = 2.0 * xc
        for k in range(2, self.K):
            t_next = (x2 * T_list[-1] - T_list[-2]).clamp(-5, 5)
            T_list.append(t_next)
        cheb = torch.stack(T_list, dim=-1).clamp(-5, 5)  # [B, D, K]
        
        # Phase-weighted: einsum
        w = self.w_real + 1j * self.w_imag
        pw = (w * torch.exp(1j * self.alpha).unsqueeze(-1)).real  # [K, C]
        out = torch.einsum('...dk,kc->...dc', cheb, pw)  # [B, D, C]
        
        # Project
        out = out.reshape(x.shape[0], -1)  # [B, D*C]
        h = F.relu(self.proj(out))
        return self.out(h)


# ============================================================
# 5. Transformer Expert — 长程依赖
# ============================================================
class TransformerExpert(Expert):
    def __init__(self, input_dim, hidden_dim=64, output_dim=4):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        h = self.proj(x).unsqueeze(1)
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a).squeeze(1)
        return self.fc(h)


# ============================================================
# 6. GNN Expert — 简化版 (MLP模拟)
# ============================================================
class GNNExpert(Expert):
    """简化: 用MLP模拟图推理 (无真实图结构时)"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=4):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# 7. AutoEncoder Expert — 异常检测
# ============================================================
class AEExpert(Expert):
    def __init__(self, input_dim, hidden_dim=64, output_dim=4):
        super().__init__(input_dim, hidden_dim, output_dim)
        latent = max(hidden_dim // 4, 8)
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent), nn.ReLU())
        self.classifier = nn.Linear(latent, output_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)


# ============================================================
# 8. Factor Expert — 简单因子模型
# ============================================================
class FactorExpert(Expert):
    def __init__(self, input_dim, hidden_dim=64, output_dim=4):
        super().__init__(input_dim, hidden_dim, output_dim)
        # Factor weights (learnable)
        self.factor_w = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        return self.factor_w(x)


# ============================================================
# Regime Router (L1)
# ============================================================
class RegimeRouter(nn.Module):
    """识别市场状态: 牛(0) / 震荡(1) / 熊(2)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_regimes: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.router = nn.Linear(hidden_dim, n_regimes)
        self.temperature = 1.0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        logits = self.router(h)
        # Gumbel-Softmax for differentiable routing
        if self.training:
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
        return probs, logits


# ============================================================
# QuantNN — 完整模型
# ============================================================
class QuantNN(nn.Module):
    """
    量化交易模型: Regime Router + Expert Pool + Fusion + Action Head
    
    Regime-Expert 映射:
      牛市(0): [CNN, LSTM, GNN, Transformer]  — 趋势跟踪
      震荡(1): [MLP, Factor, PRN, AE]          — 因子/均值回归
      熊市(2): [AE, MLP, Factor]                — 防御/对冲
    """
    
    def __init__(self, input_dim: int, n_actions: int = 4,
                 hidden_dim: int = 64, n_regimes: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.n_regimes = n_regimes
        
        # Regime Router
        self.regime_router = RegimeRouter(input_dim, hidden_dim, n_regimes)
        
        # Expert Pool
        self.experts = nn.ModuleDict({
            'cnn':         CNNExpert(input_dim, hidden_dim, n_actions),
            'lstm':        LSTMExpert(input_dim, hidden_dim, n_actions),
            'mlp':         MLPExpert(input_dim, hidden_dim, n_actions),
            'prn':         PRNExpert(input_dim, hidden_dim, n_actions, K=8, C=4),
            'transformer': TransformerExpert(input_dim, hidden_dim, n_actions),
            'gnn':         GNNExpert(input_dim, hidden_dim, n_actions),
            'ae':          AEExpert(input_dim, hidden_dim, n_actions),
            'factor':      FactorExpert(input_dim, hidden_dim, n_actions),
        })
        
        # Regime → Expert mapping
        self.regime_expert_map = {
            0: ['cnn', 'lstm', 'gnn', 'transformer'],           # Bull
            1: ['mlp', 'factor', 'prn', 'ae'],                  # Range
            2: ['ae', 'mlp', 'factor'],                          # Bear
        }
        
        # Expert selection routers (one per regime)
        self.selection_routers = nn.ModuleDict()
        for regime_id, expert_names in self.regime_expert_map.items():
            self.selection_routers[str(regime_id)] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, len(expert_names)),
            )
        
        # Load balance loss accumulator
        self._expert_usage = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
          x: [B, D] observation
        Returns:
          logits: [B, n_actions] action logits
          info: dict with routing info
        """
        B = x.shape[0]
        
        # 1. Regime routing
        regime_probs, regime_logits = self.regime_router(x)  # [B, 3]
        
        # 2. For each regime, compute expert weights and outputs
        all_logists = []
        expert_weights_all = []
        expert_usage = torch.zeros(len(self.experts), device=x.device)
        
        for regime_id in range(self.n_regimes):
            expert_names = self.regime_expert_map[regime_id]
            
            # Selection weights for this regime
            sel_logits = self.selection_routers[str(regime_id)](x)  # [B, E_r]
            sel_weights = F.softmax(sel_logits, dim=-1)  # [B, E_r]
            
            # Expert outputs
            expert_outs = []
            for name in expert_names:
                expert_outs.append(self.experts[name](x))  # [B, n_actions]
            
            expert_outs = torch.stack(expert_outs, dim=1)  # [B, E_r, n_actions]
            
            # Weighted fusion
            fused = (sel_weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # [B, n_actions]
            
            # Weight by regime probability
            regime_w = regime_probs[:, regime_id:regime_id+1]  # [B, 1]
            all_logists.append(fused * regime_w)
            
            # Track expert usage
            expert_usage_list = []
            for i, name in enumerate(expert_names):
                expert_idx = list(self.experts.keys()).index(name)
                expert_usage[expert_idx] += sel_weights[:, i].mean().item()
        
        # Combine all regimes
        action_logits = sum(all_logists)  # [B, n_actions]
        
        # Load balance loss
        expected = 1.0 / len(self.experts)
        load_balance = ((expert_usage / max(expert_usage.sum(), 1e-8) - expected) ** 2).mean()
        
        info = {
            'regime_probs': regime_probs,
            'regime_logits': regime_logits,
            'load_balance_loss': load_balance,
            'expert_usage': expert_usage.detach(),
        }
        
        return action_logits, info
    
    def act(self, x: torch.Tensor, greedy: bool = True) -> Tuple[torch.Tensor, dict]:
        """决策: 返回动作和信息"""
        with torch.no_grad():
            logits, info = self.forward(x)
            if greedy:
                action = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)
        return action, info
    
    def count_params(self) -> dict:
        counts = {}
        for name, mod in self.named_children():
            counts[name] = sum(p.numel() for p in mod.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
