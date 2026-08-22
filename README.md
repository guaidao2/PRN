# Quant-NN: 基于 PRN 的量化智能交易框架

> 谐振多项式神经网络 (Polyphase Resonance Network) + 异构专家混合 (MoE) + 世界模型

## 核心创新

### RePoN: 谐振多项式神经元

传统神经元: $y = \sigma(w^Tx + b)$  
RePoN: $y = \sum_k w_k \cdot T_k(x) \cdot e^{i\alpha_k}$

- **Chebyshev 多项式激活**: 正交基分解，天然适配周期信号
- **复数权重 + 相位**: 表达振荡模式，比实数权重更丰富
- **Hebbian 局部学习**: 不依赖反向传播的在线适应

### PRN 架构

```
Input → Phase Encoder → [Resonant Block × L] → Hyperbolic Mapper → Manifold Router → Output
```

- Resonant Block: N 个 RePoN 并行 + Complex Layer Norm + 残差
- Hyperbolic Mapper: 映射到 Poincaré 双曲圆盘
- Manifold Router: MoE 路由 (双曲距离)

## 实验结果

| 指标 | 结果 |
|------|------|
| 万能逼近 | 434 参数逼近多种非线性函数 |
| Hebbian 适应 | 任务切换后适应速度比 MLP 快 **4.5×** |
| 参数效率 | 低维时仅 MLP 的 **36%** |
| Chebyshev 正交性 | 误差 0.156，支持频率分解 |
| 推理速度 | GPU 前向传播 **3.48ms** |

## 量化应用

基于 PRN 的 QuantNN 模型在 A 股 5 分钟线回测中：

- 13 只股票, 151,008 条 K 线
- **10/13 只跑赢买入持有基准**
- 平均超额收益 **+174%**

详见 [论文](paper/PRN_Paper.md)

## 快速开始

```bash
# 安装依赖
pip install torch numpy pandas

# 运行基础实验
python prn/experiments.py
```

## 项目结构

```
├── prn/
│   ├── prn_core.py        ← PRN 核心实现 (RePoN + PhaseEncoder + ResonantBlock + HM + MR)
│   └── experiments.py     ← 基础能力实验 (逼近/适应/效率/正交性)
├── paper/
│   └── PRN_Paper.md       ← 研究论文
├── data/
│   └── stock_data.py      ← A股数据获取 (baostock)
└── examples/
    └── quant_nn_example.py ← QuantNN 模型示例
```

## 论文引用

```bibtex
@article{prn2025,
  title={PRN: Polyphase Resonance Network},
  author={coolmoon and guaidao2},
  year={2026}
}
```

## License

MIT
