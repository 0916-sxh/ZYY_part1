# 研究内容一：弛豫电压重构与多模态跨尺度融合

本文档说明研究内容一的**方法设计、代码实现、运行方式与输出物**，对应开题报告中的第一部分。

---

## 1. 研究目标

从 Zhu et al. (2022) 电池循环数据中，无监督提取每圈退化特征，并融合宏观 CC 充电时间与微观弛豫电压隐向量，得到可用于后续 RUL 预测（研究内容二、三）的**多模态跨尺度特征**。

```
原始 CSV
  → 满充后弛豫 ΔV 序列（固定 32/64 维）
  → MS-CNN 掩码自编码器（30% 掩码无监督训练）
  → 32 维弛豫隐向量 z
  → 通道注意力融合 [z, CC 宏观特征]
  → 融合特征 f（导出 .npy，模块解耦）
  → t-SNE 老化流形验证（Fig 4）
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

### 3.2 Dilated MS-CNN 掩码自编码器（MAE）

**实现文件**：`models.py` → `MSCNNMaskedAE`（编码器为 `DilatedMSConvBlock`）

| 组件 | 配置 |
|------|------|
| 编码器 | 3 层 **Hybrid DilatedMSConvBlock**（kernel 3/5/7 + dilation 2/4，残差）→ Avg+Max 池化 → Linear |
| 隐向量维度 | **32** |
| 解码器 | Linear → Conv1d 反卷积栈 → 重构 ΔV |
| 掩码比例 | **30% 连续块掩码**（随机起点的一段连续时间步，非整点随机置零） |
| 损失 | 掩码位置 MSE + **0.05×平滑正则** |

**训练策略**：
- D1 + D2 合并训练 `mae_short`（seq=32）
- D3 单独训练 `mae_long`（seq=64）
- AdamW + Cosine LR，电芯级 15% 验证集早停

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
4. 训练：SmoothL1 + AdamW(lr=6e-4)，D1 默认 **3 seed 集成**（42/43/44）取预测均值

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

### 3.6 老化流形验证（t-SNE）

**实现文件**：`figures.py` → `fig4_latent_manifold()`

- 单电芯：隐向量 t-SNE 二维映射，颜色=圈数，观察单向老化轨迹
- 全数据集：子采样对比流形结构

---

## 4. 目录结构

```
research_mae/
├── RESEARCH_CONTENT_1.md   # 本文档
├── IMPLEMENTATION.md       # Debug 与迭代记录
├── data_extract.py         # 弛豫/CC 截取 + npz 缓存
├── models.py               # MSCNNMaskedAE + ChannelAttentionFusion
├── features.py             # CC 宏观特征工程
├── train.py                # MAE + Fusion 训练
├── export_features.py      # .npy 特征导出
├── evaluate.py             # Strategy D 容量回归评估
├── figures.py              # Fig 1–7
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
| `--fusion-seeds` | 42,43,44 | D1 集成 seed |
| `--skip-train` | off | 跳过训练，加载已有权重 |

---

## 6. 输出图表

| 图 | 文件 | 内容 |
|----|------|------|
| Fig 1 | `fig1_relaxation_delta_v.png` | 10/300/600 圈 ΔV–时间曲线 |
| Fig 2 | `fig2_cc_time_dataset*.png` | CC 充电时间随圈数衰减 |
| Fig 3 | `fig3_mae_recon_*.png` | 初/中/末期：原始、30% 掩码、MS-CNN 重构 |
| Fig 4 | `fig4_latent_manifold_*.png` | 单电芯 + 全数据集 t-SNE 老化流形 |
| Fig 5 | `fig5_attention_weights.png` | 通道注意力权重随圈数变化 |
| Fig 5b | `fig5_attention_by_condition.png` | 按 C-rate 分组注意力 |
| Fig 6 | `fig6_capacity_prediction.png` | 融合特征容量预测散点（特征有效性） |
| Fig 7 | `fig7_transfer_comparison.png` | D2 迁移 / D3 留出 RMSE |

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
| MS-CNN 掩码自编码器 | ✅ `MSCNNMaskedAE` |
| 30% 连续块掩码无监督训练 | ✅ `block_mask()` |
| 编码器提取隐向量 | ✅ `encode()` |
| 静态 .npy 特征文件 | ✅ `export_features.py` |
| t-SNE 老化流形 | ✅ Fig 4 |
| CC 宏观标量 | ✅ 双通道 CC 特征 |
| 通道注意力跨尺度融合 | ✅ `GatedChannelFusion`（生产）/ `ChannelAttentionFusion`（对比） |

### 当前最佳精度（Strategy D / 留出评估）

| 数据集 | Fusion RMSE% | R² |
|--------|-------------|-----|
| D1 集成 (3 seeds) | **0.54%** | 0.992 |
| D1 单 seed | 0.69% | 0.986 |
| D2 原生留出 | **0.33%** | 0.996 |
| D3 留出 | **0.89%** | 0.990 |

详见 `output/metrics.json`、`output/RESULTS.md`。

---

## 10. 依赖

- Python 3.10+
- PyTorch、numpy、pandas、scikit-learn、matplotlib

见项目根目录 `requirements.txt` / `environment.yml`。

---

## 11. 引用数据集

Zhu et al., *Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation*, Nature Communications, 2022.
