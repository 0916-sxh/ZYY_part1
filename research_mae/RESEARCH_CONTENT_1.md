# 研究内容一：弛豫电压重构与多模态跨尺度融合

本文档说明研究内容一的**方法设计、代码实现、运行方式与输出物**，对应开题报告中的第一部分。

---

## 1. 研究目标

从 Zhu et al. (2022) 电池循环数据中，无监督提取每圈退化特征，并融合宏观 CC 充电时间与微观弛豫电压隐向量，得到可用于后续 RUL 预测（研究内容二、三）的**多模态跨尺度特征**。

```
原始 CSV
  → 满充后弛豫 ΔV 序列（固定 32/64 维）
  → Hybrid Dilated MS-CNN 掩码自编码器（30% 块掩码 + 老化轴监督）
  → 32 维弛豫隐向量 z
  → 门控通道融合 [z, CC 宏观特征]
  → 融合特征 f（导出 .npy，模块解耦）
  → 老化流形验证（Fig 4：学习老化轴 + 残差 PC）
```

---

## 2. 数据集

| 数据集 | 化学体系 | 电芯数 | 弛豫窗口 | 序列维度 |
|--------|---------|--------|---------|---------|
| Dataset 1 | NCA | 66 | 30 min | **32** |
| Dataset 2 | NCM | 55 | 30 min | **32** |
| Dataset 3 | NCM+NCA | 9 | 60 min | **64** |

每圈可用字段：弛豫电压曲线、CC 恒流充电时间、放电容量。

---

## 3. 方法细节

### 3.1 弛豫电压截取与固定维度

**实现文件**：`data_extract.py` → `find_post_charge_relaxation()`

1. 定位放电起点（`I < -50 mA`）
2. 在放电前取满足 `|control|≈0`、`|I|<5 mA`、`V>4.0 V` 的**最后一段**零电流区间（满充后 OCV 弛豫）
3. 截取固定时长（D1/D2: 30 min；D3: 60 min），均匀重采样为 **32 / 64** 个时间点
4. 计算 **ΔV = V(t) − V(t₀)**，并做全局 z-score 归一化（μ、σ 存入 cache）

### 3.2 Hybrid Dilated MS-CNN 掩码自编码器（MAE）

**实现文件**：`models.py` → `MSCNNMaskedAE`（编码器为 `DilatedMSConvBlock`）

| 组件 | 配置 |
|------|------|
| 编码器 | 3 层 **Hybrid DilatedMSConvBlock**：并行 kernel **3/5/7** + dilation **2/4**，残差连接 |
| 池化 | **AvgPool + MaxPool** 拼接（256 维）→ Linear → **32 维**隐向量 |
| 老化轴 | **aging head**：Linear → GELU → Linear → **Sigmoid**，输出 ∈ [0,1] |
| 解码器 | Linear → Conv1d 反卷积栈 → 重构 ΔV |
| 掩码比例 | **30% 连续块掩码**（随机起点的一段连续时间步） |
| 重构损失 | **0.85×掩码区 MSE + 0.15×可见区 MSE** + **0.05×平滑正则** |

**老化监督（轻量半监督，不破坏 MAE 主任务）**：

老化目标 `aging_target` 由每圈可观测量构造（`train.py` → `_aging_targets()`）：

```
aging_target = 0.55 × (1 − SOH/SOH_ref) + 0.45 × (cycle / cycle_max_cell)
```

- SOH = capacity / 标称容量（mAh）
- `cycle_max_cell` 为**该电芯内**最大圈数 → 得到**寿命比率**（跨电芯可比）
- 训练损失：`L = L_recon + λ_aging·SmoothL1(ŷ, target) + λ_rank·L_pairwise_rank`
  - `λ_aging = 0.5`，`λ_rank = 0.2`（`pairwise_ranking_loss` 保证同 batch 内老化顺序一致）
- early-stop / best checkpoint **只看重构 val MSE**（不含 aging），避免老化损失抢走解码器
- MAE 主训练结束后，额外 **40 epoch aging-head 微调**：**冻结 encoder/decoder**，仅更新 `aging_head`（干净编码 latent，无掩码）

> 说明：aging head 仅用于**塑造隐空间单调老化方向**；下游融合/RUL 仍使用 `encode()` 输出的 32 维 z，不直接使用 aging 标量。

**训练策略**：
- D1 + D2 合并训练 `mae_short`（seq=32）
- D3 单独训练 `mae_long`（seq=64）
- AdamW + Cosine LR，电芯级 15% 验证集早停，`patience=20`

训练完成后**剥离解码器**，仅用编码器对完整弛豫序列提取隐向量。

### 3.3 宏观 CC 充电时间特征

**Dataset 1 CC 突变剔除**：`cc_filter.py` 对每颗电芯 CC 时长做 rolling-median 检测，剔除突变段（约 129 圈/22506 样本），再参与融合与下游 RUL。

**实现文件**：`features.py` → `build_cc_features()`

双通道 CC 特征（反映全局退化）：
- `[0]` log(CC/CC₀)：相对首圈基线的老化衰减
- `[1]` 全局 z-score：绝对 CC 时长水平

### 3.4 跨尺度融合（门控通道注意力）

**实现文件**：`models.py` → `GatedChannelFusion`

生产训练采用 **Sigmoid 门控融合**（独立门控、无 Softmax 塌缩，验证集精度更高）。`ChannelAttentionFusion`（Softmax）仍保留于代码中，供论文方法对比。

1. 弛豫隐向量 z 经 Linear 投影 → relax_feat
2. CC 双通道特征经 MLP 嵌入 → cc_emb
3. 独立 Sigmoid 门控 g_r、g_c，归一化后加权：**f = (g_r·relax + g_c·cc) / (g_r + g_c)**
4. 训练：SmoothL1 + AdamW(lr=8e-4)，D1 默认 **5 seed 集成**（42–46）取预测均值

### 3.5 特征导出（模块解耦）

**实现文件**：`export_features.py`

每个数据集导出：

| 文件 | 形状 | 说明 |
|------|------|------|
| `features/dataset_{id}_fused.npy` | (N, 32) | **下游 RUL 主输入** |
| `features/dataset_{id}_latent.npy` | (N, 32) | 仅弛豫隐向量 |
| `features/dataset_{id}_meta.npz` | — | cell_id, cycle, capacity, cc_time, attention |
| `features/dataset_{id}_manifest.json` | — | schema 与维度说明 |

下游研究内容二/三通过 `load_fused_features(dataset_id)` 读取，无需重复 MAE 训练。

### 3.6 老化流形验证（Fig 4）

**实现文件**：`thesis_figures.py` → `fig4_latent_manifold()`

**为何不用 t-SNE 直接报告 Spearman？**  
跨电芯混画时，绝对圈数与老化阶段不对齐；t-SNE 还会把一维老化趋势打散到二维。实验表明：隐向量 PCA 第一轴与 SOH 的 |ρ| 可达 0.8+，但 t-SNE 的 |ρ| 仅 0.6–0.7。

**Fig 4 画法（三数据集 1×3 子图）**：

1. 对每数据集子采样 **3000** 点（`np.linspace` 均匀索引，非数据缺失）
2. 编码得隐向量 z，经 **aging head** 得 **Dim 1 = 学习老化轴**（∈ [0,1]）
3. 用线性回归从 z 中扣除 aging 方向，残差做 **PCA 第一主成分** → **Dim 2**
4. 颜色 = **寿命比率** `cycle / cycle_max_cell`（电芯内归一化，跨电芯可比）
5. **Spearman ρ**：Dim 1 与寿命比率的 Spearman 相关系数（取正方向）
6. **稳健显示**：按 0.5%–99.5% 分位裁剪坐标轴，避免个别 aging 离群点把主体云团压成一角（D1 曾出现此问题）

**当前结果（Strategy D 特征管线，重训后）**：

| 数据集 | Spearman ρ（Dim 1 vs 寿命比率） |
|--------|-------------------------------|
| D1 | **0.894** |
| D2 | **0.843** |
| D3 | **0.991** |

论文表述建议：*“将 MAE 学习的老化轴投影到二维流形（第二维为去老化后的残差主成分），Spearman 系数量化隐空间与寿命进程的一致性。”*

### 3.7 单电芯轨迹 + 全数据投影（Fig 11）

**实现文件**：`thesis_figures.py` → `fig11_manifold_trajectory_combo()`

**布局**：2 行 × 3 列（D1 / D2 / D3）

| 行 | 内容 |
|----|------|
| **(a) 单电芯轨迹** | 每集选取长寿命电芯，对该电芯全部隐向量做 **t-SNE**（≤800 点均匀抽样），颜色 = 圈数，灰色折线连接时间顺序 |
| **(b) 全数据老化轴** | 与 Fig 4 相同：Dim 1 = aging head，Dim 2 = 残差 PCA1，颜色 = 寿命比率 |

**叙事分工**：Fig 4 侧重跨电芯老化轴一致性；Fig 11 在同一图中对比「单电芯内连续演变轨迹」与「全数据集老化投影」，便于答辩时解释 D1/D2 二维散射 vs D3 连续流形。

---

## 4. 目录结构

```
research_mae/
├── RESEARCH_CONTENT_1.md   # 本文档
├── IMPLEMENTATION.md       # Debug 与迭代记录
├── data_extract.py         # 弛豫/CC 截取 + npz 缓存
├── models.py               # MSCNNMaskedAE + ChannelAttentionFusion
├── features.py             # CC 宏观特征工程
├── train.py                # MAE + Fusion 训练（含 aging 监督）
├── export_features.py      # .npy 特征导出
├── evaluate.py             # Strategy D 容量回归评估
├── thesis_figures.py       # 论文规格 Fig 1–5、Fig 10–11
├── figures.py              # 旧版辅助出图
├── run_all.py              # 一键入口
├── cache/                  # dataset_*.npz 数据缓存
├── checkpoints/            # mae_*.pt, fusion_*.pt
├── features/               # dataset_*_fused.npy 等（研究内容二输入）
├── figures/                # 输出图片
└── output/                 # metrics.json, 训练历史
```

---

## 5. 运行方式

```bash
conda activate battery-capacity
cd /path/to/data-driven-capacity-estimation-from-voltage-relaxation

# 完整流程：重建缓存 + MS-CNN MAE + 融合 + 导出 + 评估 + 出图
PYTHONUNBUFFERED=1 python research_mae/run_all.py --rebuild-data --device cpu

# GPU 加速（如有 CUDA）
python research_mae/run_all.py --rebuild-data --device cuda

# 已有 checkpoint，仅重新导出特征与出图
python research_mae/run_all.py --skip-train --device cpu

# 跳过 MAE，仅重训融合头
python research_mae/run_all.py --skip-mae --device cpu
```

**主要参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--rebuild-data` | off | 强制重建 npz 缓存（序列维度变更后必须） |
| `--epochs-mae` | 80 | MAE 训练轮数 |
| `--epochs-fusion` | 150 | 融合模块轮数 |
| `--fusion-seeds` | 42,43,44,45,46 | D1 集成 seed |
| `--skip-train` | off | 跳过训练，加载已有权重 |

---

## 6. 输出图表

| 图 | 文件 | 内容 |
|----|------|------|
| Fig 1 | `fig1_relaxation_voltage.png` | D1 绝对电压，每 20 圈一条，蓝→红渐变 |
| Fig 2 | `fig2_cc_time_all_datasets.png` | D1/D2/D3 CC 充电时间对比（1×3） |
| Fig 3 | `fig3_mae_reconstruction.png` | D1/D2/D3 初/中/末期：原始、30% 块掩码、重构 |
| Fig 4 | `fig4_latent_manifold.png` | 三数据集老化轴流形 + Spearman ρ |
| Fig 5 | `fig5_channel_attention.png` | 通道权重 vs 寿命比率 |
| Fig 10 | `fig10_cycle_protocol_nca_cy45.png` | NCA 整圈协议（CC/CV/静置/放电） |
| Fig 11 | `fig11_manifold_trajectory_combo.png` | 单电芯 t-SNE 轨迹 (a) + 全数据老化轴 (b) |

出图入口：`python research_mae/thesis_figures.py --device cuda`

---

## 7. 评估指标（特征有效性）

在 Dataset 1 **Strategy D** 留出电芯上报告（`evaluate.py`）：

| 方法 | 说明 |
|------|------|
| Latent Ridge | 仅弛豫隐向量 + Ridge 回归 |
| Fusion 单模型 | 通道注意力融合 + CapacityHead |
| Fusion 集成 | 3 seed 预测均值（默认） |

指标：**RMSE%**（相对标称 3.5 Ah）、**R²**。

容量回归用于验证特征质量；研究内容二将改用 **RUL** 作为预测目标。

---

## 8. 与研究内容二/三的接口

```python
from research_mae.export_features import load_fused_features

data = load_fused_features(dataset_id=1)
# data["fused"]     → (N, 32) 每圈融合特征，按 cell_id + cycle 索引
# data["cell_id"]   → 电芯标识
# data["cycle"]     → 循环号（注意非连续）
# data["capacity"]  → 放电容量 (mAh)，可用于构造 RUL 标签
```

研究内容二将对每颗电芯构造 `[f_1, …, f_i]` 序列，输入 **Quantile TCN** 预测 RUL。

---

## 9. 开题表述对齐清单

| 开题要求 | 实现状态 |
|---------|---------|
| 固定维度弛豫序列（32/64） | ✅ `data_extract.py` |
| MS-CNN / Hybrid Dilated MS-CNN MAE | ✅ `MSCNNMaskedAE` |
| 30% 连续块掩码无监督训练 | ✅ `block_mask()` |
| 老化轴轻量监督（aging head） | ✅ `aging_head` + `_aging_targets()` |
| 编码器提取隐向量 | ✅ `encode()` |
| 静态 .npy 特征文件 | ✅ `export_features.py` |
| 老化流形 + Spearman | ✅ Fig 4（老化轴投影，非 t-SNE） |
| CC 宏观标量 | ✅ 双通道 CC 特征 |
| 门控跨尺度融合 | ✅ `GatedChannelFusion` |

### 当前最佳精度（Strategy D / 留出评估）

| 数据集 | Fusion RMSE% | R² | Fig4 Spearman ρ |
|--------|-------------|-----|-----------------|
| D1 集成 (5 seeds) | **0.47%** | 0.994 | **0.894** |
| D1 单 seed | 0.52% | 0.992 | — |
| D2 原生留出 | **0.23%** | 0.998 | **0.843** |
| D3 留出 | **0.54%** | 0.996 | **0.991** |

详见 `output/metrics.json`、`output/RESULTS.md`。

---

## 10. 依赖

- Python 3.10+
- PyTorch、numpy、pandas、scikit-learn、matplotlib

见项目根目录 `requirements.txt` / `environment.yml`。

---

## 11. 引用数据集

Zhu et al., *Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation*, Nature Communications, 2022.
