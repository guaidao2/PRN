"""
PRN 基础能力实验 (优化版 — 缩小规模加速)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math

from prn_core import PRN, chebyshev_polynomials


# ============================================================
# Comparison Models
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_d, out_d, hidden=64, layers=3):
        super().__init__()
        dims = [in_d] + [hidden] * layers + [out_d]
        self.net = nn.Sequential(*[
            nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            for i in range(len(dims) - 2)
        ] + [nn.Linear(dims[-2], dims[-1])])
    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(self, in_d, out_d, dim=64, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(in_d, dim)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(dim, heads, batch_first=True),
                'ff': nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim)),
                'n1': nn.LayerNorm(dim), 'n2': nn.LayerNorm(dim),
            }))
        self.head = nn.Linear(dim, out_d)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.proj(x)
        for l in self.layers:
            a, _ = l['attn'](l['n1'](x), l['n1'](x), l['n1'](x))
            x = x + a; x = x + l['ff'](l['n2'](x))
        return self.head(x.mean(dim=1))


# ============================================================
# Exp 1: Function Approximation
# ============================================================
def experiment_1():
    print("=" * 65)
    print("实验 1: 万能逼近能力 — 函数拟合")
    print("=" * 65)

    fns = {
        "sin(3x)+cos(5x)":   lambda x: np.sin(3*x) + np.cos(5*x),
        "x²·sin(x)":         lambda x: x**2 * np.sin(x),
        "高频振荡·高斯窗":    lambda x: np.sin(12*x) * np.exp(-x**2),
    }

    for name, fn in fns.items():
        X = np.random.uniform(-3, 3, (600, 1)).astype(np.float32)
        Y = fn(X).astype(np.float32)
        Xtr, Ytr = torch.from_numpy(X[:400]), torch.from_numpy(Y[:400])
        Xte, Yte = torch.from_numpy(X[400:]), torch.from_numpy(Y[400:])

        models = {
            'MLP(64×3)':       MLP(1, 1, 64, 3),
            'Transformer':     TransformerModel(1, 1, 32, 2, 2),
            'PRN(L=2,N=4)':    PRN(1, 1, L=2, N=4, K=6, C=3, task='regression'),
        }

        print(f"\n  [{name}]")
        print(f"  {'模型':<20} {'参数':>8} {'MSE':>10} {'耗时':>8}")
        print(f"  {'-'*50}")
        for mname, model in models.items():
            opt = torch.optim.Adam(model.parameters(), lr=2e-3)
            t0 = time.time()
            for _ in range(200):
                loss = F.mse_loss(model(Xtr), Ytr)
                opt.zero_grad(); loss.backward(); opt.step()
            t1 = time.time()
            with torch.no_grad():
                mse = F.mse_loss(model(Xte), Yte).item()
            np_ = sum(p.numel() for p in model.parameters())
            print(f"  {mname:<20} {np_:>8,} {mse:>10.6f} {t1-t0:>7.1f}s")
    print()


# ============================================================
# Exp 2: Convergence Speed
# ============================================================
def experiment_2():
    print("=" * 65)
    print("实验 2: 收敛速度对比")
    print("=" * 65)

    d, out = 16, 5
    X = torch.randn(1500, d)
    Y = (X[:, :3].abs().sum(1) > 1).long() % out
    Xtr, Ytr = X[:1000], Y[:1000]
    Xte, Yte = X[1000:], Y[1000:]

    models = {
        'MLP(64×3)':      MLP(d, out, 64, 3),
        'Transformer':    TransformerModel(d, out, 64, 4, 2),
        'PRN(L=2,N=4)':   PRN(d, out, L=2, N=4, K=6, C=3, task='classification'),
    }

    epochs = 150
    print(f"\n  {'模型':<20} {'参数':>8} {'最终准确率':>10} {'达80%epoch':>12} {'耗时':>8}")
    print(f"  {'-'*62}")

    for mname, model in models.items():
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        ce = nn.CrossEntropyLoss()
        accs = []
        t0 = time.time()
        for ep in range(epochs):
            loss = ce(model(Xtr), Ytr)
            opt.zero_grad(); loss.backward(); opt.step()
            if (ep+1) % 5 == 0:
                with torch.no_grad():
                    a = (model(Xte).argmax(1) == Yte).float().mean().item()
                accs.append((ep+1, a))
        t1 = time.time()
        final_acc = accs[-1][1]
        ep80 = next((e for e, a in accs if a >= 0.8), '>150')
        np_ = sum(p.numel() for p in model.parameters())
        print(f"  {mname:<20} {np_:>8,} {final_acc*100:>9.1f}% {str(ep80):>12} {t1-t0:>7.1f}s")
    print()


# ============================================================
# Exp 3: Hebbian Online Adaptation
# ============================================================
def experiment_3():
    print("=" * 65)
    print("实验 3: Hebbian 在线适应能力")
    print("=" * 65)

    d, out = 8, 3
    def make(n, pat):
        X = torch.randn(n, d)
        if pat == 'A':   Y = (X[:, 0] + X[:, 1] > 0).long() % out
        elif pat == 'B': Y = (X[:, 2] * X[:, 3] > 0.5).long() % out
        else:            Y = ((X[:, 4] > 0) ^ (X[:, 5] > 0)).long() % out
        return X, Y

    models = {
        'MLP(64×3)':              MLP(d, out, 64, 3),
        'PRN+Hebbian':            PRN(d, out, L=2, N=4, K=6, C=3, task='classification', use_hebbian=True),
        'PRN-Hebbian':            PRN(d, out, L=2, N=4, K=6, C=3, task='classification', use_hebbian=False),
    }

    patterns = ['A', 'B', 'C']
    results = {n: [] for n in models}

    for mname, model in models.items():
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        ce = nn.CrossEntropyLoss()
        for pat in patterns:
            X, Y = make(300, pat)
            losses = []
            for ep in range(60):
                loss = ce(model(X), Y)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            with torch.no_grad():
                acc = (model(X).argmax(1) == Y).float().mean().item()
            results[mname].append({'acc': acc, 'loss5': losses[4] if len(losses)>4 else losses[-1]})

    print(f"\n  各模式准确率:")
    print(f"  {'模型':<25} {'A':>8} {'B':>8} {'C':>8}")
    print(f"  {'-'*52}")
    for mname, res in results.items():
        accs = [f"{r['acc']*100:.1f}%" for r in res]
        print(f"  {mname:<25} {accs[0]:>8} {accs[1]:>8} {accs[2]:>8}")

    print(f"\n  切换后第5个epoch loss (越低=适应越快):")
    print(f"  {'模型':<25} {'A→B':>8} {'B→C':>8}")
    print(f"  {'-'*45}")
    for mname, res in results.items():
        print(f"  {mname:<25} {res[1]['loss5']:>8.4f} {res[2]['loss5']:>8.4f}")
    print()


# ============================================================
# Exp 4: Parameter Efficiency
# ============================================================
def experiment_4():
    print("=" * 65)
    print("实验 4: 参数效率")
    print("=" * 65)

    configs = [(8, 5), (32, 10), (64, 20), (128, 50)]
    print(f"\n  {'输入→输出':<14} {'MLP':>8} {'Transformer':>12} {'PRN':>8} {'PRN/MLP':>9}")
    print(f"  {'-'*55}")
    for i, o in configs:
        ml = MLP(i,o,64,3)
        tr = TransformerModel(i,o,64,4,2)
        pr = PRN(i,o,L=2,N=4,K=6,C=3,task='classification')
        mp = sum(p.numel() for p in ml.parameters())
        tp = sum(p.numel() for p in tr.parameters())
        pp = sum(p.numel() for p in pr.parameters())
        print(f"  {i:>3}→{o:<3}       {mp:>8,} {tp:>12,} {pp:>8,} {pp/mp:>8.2f}×")
    print()


# ============================================================
# Exp 5: Chebyshev Properties
# ============================================================
def experiment_5():
    print("=" * 65)
    print("实验 5: Chebyshev 基函数分析")
    print("=" * 65)

    x = torch.linspace(-1, 1, 200).unsqueeze(-1)
    for K in [4, 8, 12]:
        T = chebyshev_polynomials(x, K).squeeze(1).detach().numpy()  # [200, K]
        dt = 2.0 / 199
        cross = []
        for i in range(min(K, 5)):
            for j in range(i+1, min(K, 5)):
                try:
                    cross.append(abs(np.trapezoid(T[:,i]*T[:,j], dx=dt)))
                except AttributeError:
                    cross.append(abs(np.trapz(T[:,i]*T[:,j], dx=dt)))
        avg = np.mean(cross)
        print(f"  K={K:>2}: 正交误差={avg:.6f}  (越接近0越好)")

    print(f"\n  Chebyshev正交基 → PRN天然擅长分解叠加周期信号")
    print(f"  金融时序=多周期叠加 → PRN的Chebyshev展开是理想选择")
    print()


# ============================================================
if __name__ == "__main__":
    print()
    print("╔" + "═"*63 + "╗")
    print("║" + " PRN 基础能力实验报告".center(55) + "║")
    print("╚" + "═"*63 + "╝")
    torch.manual_seed(42); np.random.seed(42)
    t0 = time.time()
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    experiment_5()
    print("=" * 65)
    print(f"全部完成, 总耗时: {time.time()-t0:.1f}s")
    print("=" * 65)
