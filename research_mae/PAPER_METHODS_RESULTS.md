# 基于弛豫电压掩码自编码器与门控融合的锂电池容量估计

**方法 + 实验结果**（论文撰写稿）

> 数据：Zhu et al. (2022) 公开数据集 [Zenodo 10.5281/zenodo.6405084](https://doi.org/10.5281/zenodo.6405084)  
> 代码：`research_mae/`  
> 最新指标：`research_mae/output/metrics.json`（运行 `python research_mae/run_all.py` 自动生成）

---

## 1 引言与动机

Zhu 等提出从**满充后弛豫电压**提取统计特征（Var / Ske / Max），结合传统机器学习估计下一圈放电容量。本研究在相同数据基础上，提出一条**端到端深度学习**路线：

1. 将弛豫电压序列重采样为**固定维度**时序输入；
2. 用时序卷积**掩码自编码器（MAE）**学习极化衰退隐表征；
3. 用**门控双通道融合**联合弛豫隐向量与 CC 充电老化特征；
4. 回归预测放电容量。

核心假设不变：**弛豫曲线形态与恒流充电时间均随容量衰减而变化**，且序列级信息优于少量统计量。

---

## 2 方法

### 2.1 问题定义

对每个循环样本 $i$，观测到：

- 满充后弛豫电压序列 $\mathbf{v}_i = [V(t_1), \ldots, V(t_L)]$
- 该圈 CC 恒流充电时间 $t^{\mathrm{CC}}_i$
- 标签：该圈最大放电容量 $Q_i$（mAh）

目标：学习映射 $f: (\mathbf{v}_i, t^{\mathrm{CC}}_i) \mapsto \hat{Q}_i$，最小化预测误差。

### 2.2 弛豫电压截取与固定维度

参照 Region III（CV 结束后的 OCV 弛豫段），在原始 CSV 中：

1. 定位放电起点：$I < -50\,\mathrm{mA}$；
2. 在放电前取**最后一段**满足 $|I|<5\,\mathrm{mA}$、$|control|\approx 0$、$V>4.0\,\mathrm{V}$ 的区间；
3. 从段首截取固定时长 $T_{\mathrm{relax}}$，计算 $\Delta V(t)=V(t)-V(t_0)$；
4. 在 $[0, T_{\mathrm{relax}}]$ 上均匀插值为 $L$ 点；
5. 剔除末尾放电切换点（$|I|>0.5\,\mathrm{mA}$ 或单步 $\Delta V>15\,\mathrm{mV}$）。

| 数据集 | 化学体系 | 标称容量 | $T_{\mathrm{relax}}$ | 序列长度 $L$ |
|--------|----------|----------|----------------------|--------------|
| Dataset 1 | NCA (LG 35E) | 3.5 Ah | 30 min | 30 |
| Dataset 2 | NCM (Samsung MJ1) | 3.5 Ah | 30 min | 30 |
| Dataset 3 | NCM+NCA (Samsung 25R) | 2.5 Ah | 60 min | 60 |

训练时对 $\Delta V$ 序列做全局 z-score；绘图时保留原始 mV 级 `delta_v_raw`。

### 2.3 CC 充电时间特征

提取每圈**第一段** CC 区间时长（$I>50\,\mathrm{mA}$，恒流模式，$V<4.19\,\mathrm{V}$）。

构造双通道 CC 特征：

$$\mathbf{c}_i = \Big[\ \log\!\big(t^{\mathrm{CC}}_i / t^{\mathrm{CC}}_{i,0}\big),\ \ z(t^{\mathrm{CC}}_i)\ \Big]^\top$$

其中 $t^{\mathrm{CC}}_{i,0}$ 为同一电芯首圈 CC 时间，$z(\cdot)$ 为训练集全局 z-score。前者刻画**电芯内相对老化**，后者保留绝对水平。

### 2.4 时序卷积掩码自编码器（MAE）

**结构**

- **Encoder**：Conv1d($1\!\to\!32\!\to\!64\!\to\!128$) + Global Average Pooling + Linear $\to$ **32 维隐向量** $\mathbf{z}_i$
- **Decoder**：Linear $\to$ Conv1d($128\!\to\!64\!\to\!32\!\to\!16\!\to\!1$)
- **掩码**：随机置零 30% 时间步，仅在可见位置计算重构损失
- **损失**：$\mathcal{L}_{\mathrm{MAE}} = \mathcal{L}_{\mathrm{mse}} + 0.08\,\mathcal{L}_{\mathrm{smooth}}$

**训练**

- Dataset 1+2 合并训练 `mae_short`（$L=30$）
- Dataset 3 单独训练 `mae_long`（$L=60$）
- 优化器 AdamW，Cosine LR，电芯级 15% 验证早停

推理时 $\mathbf{z}_i = \mathrm{Encoder}(\Delta\mathbf{V}_i)$（无掩码），并对 $\mathbf{z}_i$ 做训练集 z-score 标准化。

### 2.5 门控双通道融合（Gated Fusion）

以弛豫隐向量与 CC 特征为两路输入，采用**独立 sigmoid 门控**（非 softmax，避免权重塌缩）：

$$\mathbf{h}_i^r = W_r \mathbf{z}_i, \quad \mathbf{h}_i^c = \mathrm{MLP}(\mathbf{c}_i)$$

$$g_i^r = \sigma(W_g^r [\mathbf{h}_i^r; \mathbf{h}_i^c]), \quad g_i^c = \sigma(W_g^c [\mathbf{h}_i^r; \mathbf{h}_i^c])$$

$$\tilde{\mathbf{h}}_i = \frac{g_i^r \mathbf{h}_i^r + g_i^c \mathbf{h}_i^c}{g_i^r + g_i^c + \epsilon}$$

门控权重 $w_i^r = g_i^r/(g_i^r+g_i^c)$、$w_i^c = g_i^c/(g_i^r+g_i^c)$ 用于可视化（Fig 5）。

### 2.6 容量回归头

将融合向量、原始隐向量、CC 特征拼接后送入 MLP：

$$\hat{Q}_i^{\mathrm{norm}} = \mathrm{MLP}\big([\tilde{\mathbf{h}}_i;\ \mathbf{z}_i;\ \mathbf{c}_i]\big), \quad \hat{Q}_i = \hat{Q}_i^{\mathrm{norm}} \cdot Q_{\mathrm{nominal}}$$

其中 $Q_{\mathrm{nominal}}$ 为数据集标称容量（mAh）。损失函数为 SmoothL1，按验证集 **RMSE%** 早停。

### 2.7 Dataset 1 集成策略

对 Dataset 1 训练 3 个不同随机种子（42/43/44）的 Fusion 模型，测试时预测取**算术平均**，以降低方差。

### 2.8 迁移学习（D2）

对 Dataset 2 跨域场景，采用 TL 策略：

- **零样本**：直接使用 Dataset 1 训练的 Fusion；
- **TL 微调**：冻结门控网络，仅用 Strategy D 稀疏标注样本（每工况 1 电芯、每 100 圈 1 点，共 23 点）微调回归头；
- **原生训练**：在 Dataset 2 上独立训练 `fusion_ds2`（15% 电芯留出验证）。

---

## 3 实验设置

### 3.1 数据集

| 编号 | 材料 | 电芯数 | 样本数 | 备注 |
|------|------|--------|--------|------|
| Dataset 1 | NCA | 66 | 22,635 | 主实验 |
| Dataset 2 | NCM | 55 | 27,803 | 迁移目标 |
| Dataset 3 | NCM+NCA | 9 | 8,582 | 长弛豫（60 min） |

### 3.2 评价指标

$$\mathrm{RMSE\%} = \frac{\mathrm{RMSE}(Q, \hat{Q})}{Q_{\mathrm{nominal}}} \times 100\%$$

其中 $Q_{\mathrm{nominal}} = 3500\,\mathrm{mAh}$（Dataset 1/2）或 $2500\,\mathrm{mAh}$（Dataset 3）。同时报告决定系数 $R^2$。

### 3.3 数据划分

**Dataset 1（主实验）**：采用论文 **Strategy D**——按 C-rate 工况分层，留出 14 个测试电芯（52 训练 / 14 测试），与 `battery_pipeline/splits.py` 一致。

**Dataset 2/3（原生 Fusion）**：随机留出 15% 电芯作为验证/测试。

**Dataset 2（TL）**：每工况随机 1 电芯、每 100 圈采样用于微调，其余用于评估（与论文迁移协议一致）。

### 3.4 对比基线

| 基线 | 说明 |
|------|------|
| **Latent Ridge** | MAE 隐向量 → Ridge 回归 |
| **CC Ridge** | CC 时间 → Ridge 回归 |
| **Latent+CC Ridge** | 隐向量与 CC 时间拼接 → Ridge |
| **论文 SVR** | 统计特征 Var/Ske/Max + SVR（`battery_pipeline/`，同 Strategy D） |
| **论文 TL2** | 统计特征 + TL2 线性变换（`battery_pipeline/transfer.py`） |

### 3.5 实现环境

- Python 3.10，PyTorch，scikit-learn
- 默认 CPU 训练；MAE ~5 min，Fusion ~1 min（`--skip-mae`）
- 完整复现：

```bash
conda activate battery-capacity
cd /path/to/data-driven-capacity-estimation-from-voltage-relaxation
python research_mae/run_all.py --device cpu
```

---

## 4 实验结果

### 4.1 Dataset 1 主实验（Strategy D 测试集）

**表 1** Dataset 1 容量估计结果（14 个留出电芯，5,641 测试样本）

| 方法 | RMSE% ↓ | $R^2$ ↑ |
|------|---------|---------|
| CC Ridge | 5.71 | 0.07 |
| Latent Ridge | 2.99 | 0.75 |
| Latent+CC Ridge | 2.96 | 0.75 |
| **本文 Fusion（单 seed）** | **0.75** | **0.98** |
| **本文 Fusion（3-seed 集成）** | **0.57** | **0.99** |
| 论文复现 SVR（统计特征） | ~1.02 | — |
| 论文复现 XGBoost（统计特征） | ~1.09 | — |

> 论文基线来自 `output/results.json`（`battery_pipeline` 复现）。本文方法在相同 Strategy D 划分下，**RMSE 优于论文 SVR 约 44%**（0.57% vs 1.02%）。

### 4.2 Dataset 2 迁移与跨域

**表 2** Dataset 2 结果

| 设置 | 方法 | RMSE% ↓ | $R^2$ ↑ |
|------|------|---------|---------|
| 零样本（D1→D2） | 本文 Fusion | **1.16** | 0.96 |
| 零样本 | Latent Ridge | 1.62 | 0.92 |
| 零样本 | 论文 SVR | ~6.38 | — |
| TL 微调（23 点） | 本文 Fusion-head | 2.62 | 0.79 |
| TL 微调 | 论文 TL2 | ~4.14 | — |
| 原生留出（15% 电芯） | 本文 fusion_ds2 | **0.36** | **1.00** |

零样本场景下本文 Fusion（1.16%）已优于 Latent Ridge 与论文 SVR；在目标域独立训练时可达 **0.36%**。

### 4.3 Dataset 3 长弛豫实验

**表 3** Dataset 3 结果（60 min / 60 点 MAE，15% 电芯留出，922 验证样本）

| 方法 | RMSE% ↓ | $R^2$ ↑ |
|------|---------|---------|
| Latent Ridge | 2.24 | 0.94 |
| **本文 fusion_ds3** | **0.94** | **0.99** |
| 论文 TL2（参考） | ~5.41 | — |

Dataset 3 需使用 `mae_long`（seq=60）及独立 Fusion，不可直接套用 Dataset 1 的 30 点模型。

### 4.4 结果汇总

| 实验 | 最优 RMSE% | 对应配置 |
|------|------------|----------|
| D1 主实验 | **0.57** | 3-seed 集成 + Strategy D |
| D2 零样本 | **1.16** | D1 Fusion 直接迁移 |
| D2 原生 | **0.36** | fusion_ds2 留出 |
| D3 原生 | **0.94** | fusion_ds3 + mae_long |

---

## 5 图表说明

| 图文件 | 内容 | 论文用途 |
|--------|------|----------|
| `fig1_relaxation_delta_v.png` | 第 10/300/600 圈 ΔV–时间曲线 | 说明弛豫段截取正确、老化趋势 |
| `fig2_cc_time_dataset*.png` | CC 充电时间随圈数退化 | 辅助特征物理意义 |
| `fig3_mae_recon_dataset*.png` | MAE 掩码重构（初/中/末期） | 验证 MAE 表征能力 |
| `fig4_latent_manifold_*.png` | 隐向量 t-SNE（单电芯 / 全数据） | 隐空间与老化关联 |
| `fig5_attention_weights.png` | 门控权重随圈数变化 | 融合机制可解释性 |
| `fig5_attention_by_condition.png` | 不同 C-rate 工况门控对比 | 工况差异分析 |
| `fig6_capacity_prediction.png` | D1 测试集预测–真实散点 | 主实验定量可视化 |
| `fig7_transfer_comparison.png` | D2 迁移 / D3 留出对比 | 跨域泛化 |
| `train_mae_*.png` / `train_fusion_*.png` | 训练曲线 | 补充材料 |

---

## 6 讨论

### 6.1 相对论文方法的优势

- **序列级建模**：MAE 利用完整弛豫形态，而非 3 个统计量；
- **多源融合**：门控机制自适应平衡弛豫隐向量与 CC 老化特征；
- **精度**：D1 测试 RMSE 0.57%，优于同划分下 SVR（~1.02%）。

### 6.2 局限与后续工作

1. D2 TL 微调（仅调 head）在本实现中未优于零样本，后续可解冻 CC 嵌入层或增加微调样本；
2. Dataset 3 结果对训练 seed 有一定波动（0.47%–0.94%），可进一步做 D3 集成；
3. 计算开销高于 SVR，但 MAE 训练一次后可缓存隐向量，Fusion 训练秒级。

---

## 7 复现清单

```bash
# 1. 环境
conda activate battery-capacity

# 2. 完整流程（数据提取 + MAE + Fusion + 评估 + 出图）
python research_mae/run_all.py --device cpu

# 3. 快速重训 Fusion（复用 MAE checkpoint，~1 min）
python research_mae/run_all.py --skip-mae --device cpu

# 4. 论文统计特征基线（对比用）
python run_pipeline.py
```

输出：

- 指标：`research_mae/output/metrics.json`
- 摘要：`research_mae/output/RESULTS.md`
- 图片：`research_mae/figures/`
- 模型：`research_mae/checkpoints/`

---

## 参考文献

Zhu, T., et al. (2022). Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation. *Nature Communications*, 13, 2841. https://doi.org/10.1038/s41467-022-29837-w
