# FoundAD 全局感知主分支设计方案

## 一、背景与动机

### 1.1 当前 FoundAD 的局限

当前 FoundAD 的异常检测完全基于 **patch-level** 的 MSE 比对：

- 每个 patch 独立判断是否异常
- 通过 top-K 聚合得到图像级异常分数
- 无法感知"组件缺失"、"排列错误"等**逻辑异常**
- 对空间变换敏感（patch 位置固定）

### 1.2 主分支定位

主分支应作为一个**具有全局感知能力的模块**：

- 有效感受逻辑异常（组件数量、空间关系、排列规则）
- 对空间变换不敏感（排列不变性）
- 与现有 patch 分支互补，而非替代

---

## 二、方案一：GeM Pooling（快速验证基线）

### 2.1 来源

**论文**: Fine-tuning CNN Image Retrieval with No Human Annotation  
**作者**: Filip Radenović, Giorgos Tolias, Ondřej Chum  
**发表**: TPAMI 2018  
**arXiv**: https://arxiv.org/abs/1711.02512  
**本地路径**: `papers/global_branch/GeM_Pooling_TPAMI2018.pdf`

### 2.2 核心思想

GeM (Generalized Mean) Pooling 是一种可学习的特征聚合方法，通过引入可调参数 $p$，在平均池化和最大池化之间实现平滑过渡：

$$g = \left(\frac{1}{N}\sum_{i=1}^{N} z_i^{p}\right)^{1/p}$$

- 当 $p=1$：退化为平均池化（关注整体统计分布）
- 当 $p \to \infty$：退化为最大池化（关注最显著特征）
- **$p$ 可学习**：模型自适应地在两者之间找到最优平衡

### 2.3 对 FoundAD 的适配

```
Image → DINOv2 Encoder → Patch Features z ∈ R^{B×N×D}
                              │
                    ┌─────────┴──────────┐
                    ↓                    ↓
              Patch Predictor        GeM Pooling
              (已有，不改动)          (新增)
                    ↓                    ↓
              p ∈ R^{B×N×D}         g ∈ R^{B×D}
                    ↓                    ↓
              L_patch = MSE(z,p)    Global Predictor (MLP)
                                         ↓
                                   g_pred ∈ R^{B×D}
                                         ↓
                                   L_global = MSE(g, g_pred)
                                         ↓
              Anomaly Score = L_patch + λ × L_global
```

### 2.4 优势

| 特性 | 说明 |
|------|------|
| 排列不变性 | ✅ 对称函数，对 patch 顺序不敏感 |
| 空间鲁棒性 | ✅ 天然的空间变换鲁棒性 |
| 实现复杂度 | 极低，仅需 ~10 行代码 |
| 参数量 | 仅 1 个可学习参数 $p$ |
| 可解释性 | $p$ 值反映模型偏好（mean vs max） |

### 2.5 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| $p$ 初始化 | 3.0 | 偏向 max pooling，关注显著特征 |
| $\lambda$ | 0.1 | 全局损失权重，从 0.05~0.2 搜索 |
| Global Predictor 深度 | 2 层 MLP | 轻量，避免 few-shot 过拟合 |

---

## 三、方案二：Perceiver-style Learnable Tokens（主力方案）

### 3.1 来源

**论文 1**: Perceiver: General Perception with Iterative Attention  
**作者**: Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, Joao Carreira  
**发表**: ICML 2021  
**arXiv**: https://arxiv.org/abs/2103.03206  
**本地路径**: `papers/global_branch/Perceiver_ICML2021.pdf`

**论文 2**: Perceiver IO: A General Architecture for Structured Inputs & Outputs  
**作者**: Andrew Jaegle et al.  
**发表**: ICLR 2022  
**arXiv**: https://arxiv.org/abs/2107.14795  
**本地路径**: `papers/global_branch/Perceiver_IO_ICLR2022.pdf`

### 3.2 核心思想

Perceiver 的核心创新是**非对称注意力机制**：

1. 引入一组固定数量、可学习的 **latent tokens**（潜在变量）
2. 通过 **Cross-Attention** 让 latent tokens 主动从输入中"查询"信息
3. 通过 **Self-Attention** 让 latent tokens 之间交互，避免冗余
4. 迭代精炼：多层 cross-attention + self-attention 逐步提取全局信息

**关键公式**（Cross-Attention）：

$$\text{CrossAttn}(Q_{latent}, K_{input}, V_{input}) = \text{softmax}\left(\frac{Q_{latent} K_{input}^T}{\sqrt{d}}\right) V_{input}$$

其中 $Q_{latent} \in \mathbb{R}^{K \times D}$（K 个 latent tokens），$K_{input}, V_{input} \in \mathbb{R}^{N \times D}$（N 个 patch features），$K \ll N$。

**复杂度**：$O(K \cdot N \cdot D)$，与输入规模线性相关。

### 3.3 对 FoundAD 的适配

```
Image → DINOv2 Encoder → Patch Features z ∈ R^{B×N×D}
                              │
                    ┌─────────┴──────────────┐
                    ↓                        ↓
              Patch Predictor          Global Perceiver
              (已有，不改动)            (新增)
                    ↓                        ↓
              p ∈ R^{B×N×D}           ┌─────────────────┐
                    ↓                 │ Latent Tokens    │
              L_patch = MSE(z,p)      │ Q ∈ R^{B×K×D}   │
                                      │      ↓           │
                                      │ Cross-Attn(z)   │
                                      │      ↓           │
                                      │ Self-Attn       │
                                      │      ↓           │
                                      │ G ∈ R^{B×K×D}   │
                                      └─────────────────┘
                                               ↓
                                        Global Predictor
                                               ↓
                                        G_pred ∈ R^{B×K×D}
                                               ↓
                                        L_global = MSE(G, G_pred)
                                               ↓
              Anomaly Score = L_patch + λ × L_global
```



### 3.5 优势

| 特性 | 说明 |
|------|------|
| 排列不变性 | ✅ Cross-attention 对 key/value 顺序不敏感 |
| 多视角感知 | K 个 token 可自发学会关注不同方面（数量、布局、纹理等） |
| 可解释性 | 可可视化每个 token 的 attention map |
| 自适应聚焦 | 不同于固定池化，token 学会"看哪里" |
| 可扩展 | K 是可调超参数 |

### 3.6 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| K (latent tokens 数量) | 4 | 起点，4 个 token 分别关注不同全局属性 |
| Cross-Attention 层数 | 2 | 允许迭代精炼 |
| Self-Attention 层数 | 1 | token 间交互，避免冗余 |
| $\lambda$ | 0.1 | 全局损失权重 |
| Global Predictor 深度 | 2 层 MLP | 轻量 |

---

## 四、方案三：Slot Attention — 竞争性组件分解（逻辑异常专项）

### 4.1 来源

**论文**: Object-Centric Learning with Slot Attention  
**作者**: Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, Thomas Kipf  
**发表**: NeurIPS 2020  
**arXiv**: https://arxiv.org/abs/2006.15055  
**本地路径**: `papers/global_branch/Slot_Attention_NeurIPS2020.pdf`

### 4.2 核心思想

Slot Attention 的核心创新是**竞争性注意力 (Competitive Attention)**——与 Perceiver 的关键区别在于 **softmax 的方向**：

| | Perceiver Cross-Attention | Slot Attention |
|---|---|---|
| Softmax 方向 | 对 **keys** 做 softmax | 对 **slots (queries)** 做 softmax |
| 行为 | 每个 slot 独立查询输入 | slots 之间**互相竞争**绑定 |
| 结果 | 重叠、模糊的绑定 | 互斥、清晰的分解 |
| 类比 | 每个人各自看全场 | 每个人抢不同的地盘 |

**数学对比**：

- **Perceiver（标准 cross-attention）**：$\text{attn}_{ij} = \text{softmax}_{\color{red}{j}}\left(\frac{Q_i \cdot K_j^T}{\sqrt{d}}\right)$
- **Slot Attention（竞争性 attention）**：$\text{attn}_{ij} = \text{softmax}_{\color{red}{i}}\left(\frac{Q_i \cdot K_j^T}{\sqrt{d}}\right)$

注意 softmax 的下标：Perceiver 在 key 维度做 softmax（每个 query 独立归一化），Slot Attention 在 **slot 维度**做 softmax（每个 key 位置被多个 slot 竞争）。

**迭代精炼流程**：

```
初始化 slots ~ N(μ, σ²)
  ↓
for t in 1..T:
  attn = softmax_slots(Q(slots) · K(inputs)^T / √d)   ← 竞争！
  updates = attn · V(inputs)
  slots = GRU(slots, updates)                           ← 迭代更新
  slots = slots + MLP(LayerNorm(slots))                 ← 残差精炼
```

### 4.3 为什么 Slot Attention 对逻辑异常检测特别有价值

#### 4.3.1 天然的组件级分解

Slot Attention 的核心行为是 **slots 竞争绑定到不同区域**。对于工业产品图像：

```
正常图像 (bottle):
  Slot 0 → 瓶身 (body)
  Slot 1 → 瓶盖 (cap)  
  Slot 2 → 标签 (label)
  Slot 3 → 背景 (background)

逻辑异常 (缺瓶盖):
  Slot 0 → 瓶身
  Slot 1 → ??? (无匹配，attention 分散 → 高异常)
  Slot 2 → 标签
  Slot 3 → 背景
```

#### 4.3.2 排列/数量感知

| 异常类型 | Slot Attention 行为 | 检测信号 |
|---------|-------------------|---------|
| 组件缺失 | 某个 slot 找不到匹配 | attention 熵高 → 异常 |
| 多余组件 | 多个 slot 竞争同一区域 | attention 冲突 → 异常 |
| 组件错位 | slot 绑定位置偏移 | 与正常模式不同 → 异常 |
| 组件破损 | slot 绑定区域特征异常 | predictor 误差大 → 异常 |

#### 4.3.3 与 Perceiver 的场景互补

| 场景 | Perceiver 更好 | Slot Attention 更好 |
|------|:---:|:---:|
| 纹理异常（划痕、污点） | ✅ 全局特征偏差 | ❌ 局部变化不影响 slot 绑定 |
| 组件缺失 | ❌ 难以感知 | ✅ slot 无匹配 → 高异常 |
| 组件错位 | ❌ 排列不变性反而有害 | ✅ 绑定位置变化可检测 |
| 多余组件 | ❌ 难以感知 | ✅ slot 竞争冲突 |

### 4.4 对 FoundAD 的适配

```
Image → DINOv2 Encoder → Patch Features z ∈ R^{B×N×D}
                              │
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
        Patch Predictor   Perceiver       Slot Attention
        (已有)            (全局摘要)      (组件分解)
              ↓               ↓               ↓
        L_patch          L_global_1      L_global_2
                             │               │
                             └───────┬───────┘
                                     ↓
                           融合异常分数
```

**训练信号设计**（关键创新点）：

原始 Slot Attention 依赖重建损失（decode slots → image），但 FoundAD 不做重建。替代方案：

**方案 A：Slot Prediction Loss（类比 patch predictor）**
```python
slots = slot_attention(patch_features)  # [B, K, D]
slots_pred = slot_predictor(slots)       # [B, K, D]
L_slot = MSE(slots, slots_pred)
```
- 正常样本：predictor 学会预测 slot 的"正常绑定模式"
- 异常样本：slot 绑定异常 → predictor 预测偏差大

**方案 B：Slot Consistency Loss（跨样本一致性）**
```python
# 同一类别的正常样本，slot 绑定模式应该一致
slots_1 = slot_attention(features_1)
slots_2 = slot_attention(features_2)
# 用 Hungarian matching 对齐 slots
L_consistency = MSE(aligned(slots_1), aligned(slots_2))
```

### 4.5 核心模块伪代码

```python
class CompetitiveSlotAttention(nn.Module):
    """
    Slot Attention for anomaly detection.
    关键区别：softmax 在 slot 维度，而非 key 维度。
    """
    def __init__(self, embed_dim=768, num_slots=4, num_iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.num_iters = num_iters
        
        # 可学习的 slot 初始分布参数
        self.slots_mu = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # GRU 用于迭代更新 slots
        self.gru = nn.GRUCell(embed_dim, embed_dim)
        
        # LayerNorm + MLP
        self.norm_inputs = nn.LayerNorm(embed_dim)
        self.norm_slots = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        
    def forward(self, inputs):
        """
        inputs: [B, N, D] patch features
        returns: [B, K, D] slot representations
        """
        B, N, D = inputs.shape
        inputs = self.norm_inputs(inputs)
        
        # 随机初始化 slots
        slots = self.slots_mu + torch.randn(
            B, self.num_slots, D, device=inputs.device
        ) * torch.exp(self.slots_logsigma)
        
        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            q = self.q_proj(slots)    # [B, K, D]
            k = self.k_proj(inputs)   # [B, N, D]
            v = self.v_proj(inputs)   # [B, N, D]
            
            # ★ 关键：softmax 在 slot 维度 (dim=1)，实现竞争
            attn = torch.einsum('bkd,bnd->bkn', q, k) * (D ** -0.5)
            attn = F.softmax(attn, dim=1)  # [B, K, N] — 沿 K 竞争！
            
            updates = torch.einsum('bkn,bnd->bkd', attn, v)  # [B, K, D]
            
            # GRU 更新
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, self.num_slots, D)
            
            # MLP + residual
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots
```

### 4.6 潜在问题与对策

| 问题 | 原因 | 对策 |
|------|------|------|
| Few-shot 下 slot 绑定不稳定 | 需要大量样本学习稳定绑定 | 共享初始化 + 位置先验 + 温度退火 |
| 训练信号设计 | 原始依赖重建损失 | 用 slot prediction loss 替代 |
| 计算开销 | 3 轮迭代 × GRU | 减少到 2 轮（简单工业场景足够） |
| Slot 坍缩 | 所有 slot 绑定同一区域 | 竞争机制天然防止（softmax 在 slot 维度） |

### 4.7 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| K (slots 数量) | 4 | 对应典型工业产品的组件数 |
| 迭代次数 T | 2~3 | 2 轮对简单场景足够 |
| Slot Predictor 深度 | 2 层 MLP | 轻量 |
| $\lambda_{slot}$ | 0.1 | slot 损失权重 |
| 温度退火 | 1.0 → 0.5 | 训练中逐步硬化 attention |

---

## 五、方案对比

| 维度 | GeM Pooling | Perceiver | Slot Attention |
|------|:---:|:---:|:---:|
| 排列不变性 | ✅ 天然 | ✅ 天然 | ⚠️ 部分（绑定位置可检测） |
| 组件分解能力 | ❌ | ❌ | ✅ **核心优势** |
| 表达能力 | 低（单向量） | 高（K 个向量） | 高（K 个向量 + 竞争） |
| 逻辑异常检测 | 弱 | 中 | **强** |
| 纹理异常检测 | 中 | 强 | 弱 |
| 可解释性 | 低 | 高（attention map） | **最高**（竞争 attention + 绑定可视化） |
| 参数量 | 1 个 (p) | ~K×D×4 | ~K×D×4 + GRU |
| 计算量 | O(N×D) | O(K×N×D) | O(T×K×N×D) |
| Few-shot 稳定性 | 高 | 中 | 低（需特殊处理） |
| 实现复杂度 | 极低 | 中 | 中高 |
| 推荐阶段 | 快速验证 | 正式方案 | 逻辑异常专项 |

---

## 六、实施路线

```
阶段 1: GeM Pooling（快速验证）
  ├── 实现简单，1-2 小时可跑通
  ├── 验证全局分支是否提升逻辑异常检测
  ├── 在 MVTec-AD 上小规模验证（100~300 epoch）
  └── 作为 baseline

阶段 2: Perceiver-style Learnable Tokens（主力方案）
  ├── K=4, 2 层 cross-attn, 1 层 self-attn
  ├── 可视化每个 token 的 attention map
  ├── 对比 GeM baseline
  └── 在 MVTec LOCO 上评估逻辑异常

阶段 3: Slot Attention（逻辑异常专项）
  ├── K=4, T=2~3 轮迭代
  ├── 用 slot prediction loss 替代重建损失
  ├── 可视化 slot 竞争绑定（attention map）
  ├── 重点评估：组件缺失、错位、多余等逻辑异常
  └── 与 Perceiver 对比，分析互补性

阶段 4: 进阶优化
  ├── 多尺度 cross-attn（不同 encoder 层）
  ├── Token/Slot 多样性正则（防止冗余）
  ├── Perceiver + Slot Attention 融合
  └── 与 patch 分支的交互（双向信息流）
```

---

## 七、相关论文索引

| 论文 | 发表 | 核心贡献 | 本地路径 |
|------|------|---------|---------|
| GeM Pooling | TPAMI 2018 | 可学习广义均值池化 | `papers/global_branch/GeM_Pooling_TPAMI2018.pdf` |
| Perceiver | ICML 2021 | 非对称注意力 + latent bottleneck | `papers/global_branch/Perceiver_ICML2021.pdf` |
| Perceiver IO | ICLR 2022 | 通用结构化输入输出架构 | `papers/global_branch/Perceiver_IO_ICLR2022.pdf` |
| Slot Attention | NeurIPS 2020 | 竞争性注意力 + 对象中心学习 | `papers/global_branch/Slot_Attention_NeurIPS2020.pdf` |
| AnomalyMoE | arXiv 2025 | 三层专家：patch/component/global | 待下载 |
| SALAD | ICCV 2025 | Composition map 分布建模 | 待下载 |
| PUAD | 2024 | Picturable vs Unpicturable 异常分离 | 待下载 |
| Set Transformer | ICML 2019 | ISAB + PMA 排列不变聚合 | 待下载 |
| ViT Needs Registers | ICLR 2024 | Register tokens 吸收 artifact | 待下载 |
