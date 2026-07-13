# 融合弛豫电压多尺度特征的锂电池剩余寿命预测

本仓库基于 Zhu et al. (2022) 公开数据集，实现**论文复现**与两项研究内容：

| 模块 | 目录 | 内容 |
|------|------|------|
| 论文复现 | `battery_pipeline/` | 统计特征 + SVR/XGBoost/TL2 |
| **研究内容一** | `research_mae/` | MS-CNN MAE + 门控融合 → 退化特征 `.npy` |
| **研究内容二** | `research_rul/` | Quantile TCN + Pinball Loss + 单调惩罚 → RUL |

```
研究内容一（特征）  →  融合特征 .npy
        ↓
研究内容二（RUL）  →  Quantile TCN 预测剩余寿命 + 不确定性区间
```

原始实验数据：[Zenodo 10.5281/zenodo.6405084](https://doi.org/10.5281/zenodo.6405084)

---

## 项目结构

```
.
├── Dataset_{1,2,3}_*/          # 原始循环 CSV（需自行下载）
├── battery_pipeline/           # 论文复现
├── research_mae/               # 研究内容一
│   ├── run_all.py              # MAE + 融合 + 特征导出 + Fig 1–5
│   ├── thesis_figures.py       # 论文规格图表
│   ├── cc_filter.py            # D1 CC 突变剔除
│   ├── features/               # dataset_*_fused.npy
│   ├── figures/                # Fig 1–5
│   └── RESEARCH_CONTENT_1.md   # 详细说明
├── research_rul/               # 研究内容二
│   ├── run_all.py              # Quantile TCN + Fig 6–9
│   ├── figures/                # Fig 6–9
│   └── RESEARCH_CONTENT_2.md   # 详细说明
├── run_research.py             # 研究内容一 + 二 一键运行
├── run_pipeline.py             # 论文复现入口
├── run_figures.py              # 论文图表
├── environment.yml
└── requirements.txt
```

---

## 环境配置

```bash
conda env create -f environment.yml
conda activate battery-capacity
# 或：pip install -r requirements.txt
```

主要依赖：`numpy`、`pandas`、`scikit-learn`、`torch`、`matplotlib`、`scipy`。

---

## 一键运行（研究内容一 + 二）

```bash
conda activate battery-capacity
cd /path/to/data-driven-capacity-estimation-from-voltage-relaxation

# 完整流程：数据缓存 → MAE → 融合 → 特征导出 → RUL 训练 → 全部图表
python run_research.py --rebuild-data --device cpu
```

分步运行：

```bash
# 仅研究内容一
python research_mae/run_all.py --rebuild-data --device cpu

# 仅研究内容二（需先完成特征导出）
python research_rul/run_all.py --device cpu

# 仅重新出 Fig 6–9
python research_rul/run_all.py --figures-only --device cpu
```

---

## 研究内容一：退化特征提取

### 方法

```
原始 CSV
  → 满充后弛豫 ΔV 序列（D1/D2: 32 点，D3: 64 点）
  → MS-CNN 掩码自编码器（30% 掩码无监督）→ 32 维隐向量
  → 门控通道融合 [弛豫隐向量, CC 充电时间]
  → 融合特征 f → 导出 .npy（模块解耦）
```

**Dataset 1 特殊处理**：`cc_filter.py` 对 CC 充电时间做 rolling-median 突变检测，剔除异常段（约 129 圈）。

### 定量结果（Strategy D 留出电芯，容量回归验证特征质量）

| 方法 | Test RMSE% | R² |
|------|------------|-----|
| **D1 集成（3 seeds）** | **0.55%** | 0.992 |
| D1 单模型 | 0.67% | 0.986 |
| 论文 SVR 基线 | ~1.02% | — |
| D2 原生留出 | **0.33%** | 0.996 |
| D3 原生留出 | 0.89% | 0.990 |

### 图表（Fig 1–5）

#### Fig 1 — 弛豫电压提取（Dataset 1，绝对电压，老化渐变）

![Fig 1](research_mae/figures/fig1_relaxation_voltage.png)

#### Fig 2 — 恒流充电时间退化（Dataset 1，CC 突变已剔除）

![Fig 2](research_mae/figures/fig2_cc_time_dataset1.png)

#### Fig 3 — MS-CNN MAE 掩码重构（D1 / D2 / D3，初·中·末期）

![Fig 3](research_mae/figures/fig3_mae_reconstruction.png)

#### Fig 4 — 隐向量老化流形（t-SNE，分数据集 + Spearman）

![Fig 4](research_mae/figures/fig4_latent_manifold.png)

#### Fig 5 — 通道注意力权重 vs 归一化寿命比率

![Fig 5](research_mae/figures/fig5_channel_attention.png)

详细说明：[research_mae/RESEARCH_CONTENT_1.md](research_mae/RESEARCH_CONTENT_1.md)

---

## 研究内容二：RUL 预测 + 不确定性量化

### 方法

```
融合特征序列 [f_1, …, f_i]
  → Quantile TCN（因果卷积，5%/50%/95% 分位数）
  → Pinball Loss + 单调递减物理惩罚
  → RUL 点预测 + 90% 置信区间
```

- **RUL 标签**：EOL = 80% 标称容量，RUL_i = N_EOL − cycle_i；删失电芯剔除
- **评估划分**：Dataset 1 Strategy D 留出 14 颗测试电芯

### 定量结果（Strategy D 测试集）

| 指标 | 数值 |
|------|------|
| **RMSE** | **24.3 圈** |
| **MAE** | 13.7 圈 |
| PICP (90%) | 0.49 |
| PINAW | 0.13 |

### 消融实验（Fig 7）

| 特征输入 | RMSE (圈) | MAE (圈) |
|---------|-----------|----------|
| 仅弛豫 latent | 37.0 | 20.7 |
| 仅 CC | 29.0 | 15.4 |
| 拼接 concat | 32.4 | 18.2 |
| **融合 fused（本文）** | **24.1** | **13.3** |

### 图表（Fig 6–9）

#### Fig 6 — 单调物理惩罚有效性（有/无约束 RUL 预测对比）

![Fig 6](research_rul/figures/fig6_monotonic_penalty.png)

#### Fig 7 — 多特征融合消融（RMSE / MAE）

![Fig 7](research_rul/figures/fig7_ablation.png)

#### Fig 8 — 全寿命 RUL 预测与 90% 置信区间

![Fig 8](research_rul/figures/fig8_rul_confidence.png)

#### Fig 9 — 跨数据集零样本迁移（D1 训练 → D2 / D3）

| Dataset 2 (NCM) | Dataset 3 (NCM+NCA) |
|-----------------|---------------------|
| ![Fig 9 D2](research_rul/figures/fig9_transfer_dataset2.png) | ![Fig 9 D3](research_rul/figures/fig9_transfer_dataset3.png) |

详细说明：[research_rul/RESEARCH_CONTENT_2.md](research_rul/RESEARCH_CONTENT_2.md)

---

## 论文复现（统计特征 + 经典 ML）

从满充后弛豫电压提取 `[Var, Ske, Max]`，用 ElasticNet / XGBoost / SVR 估计容量；Dataset 2/3 采用 TL2 迁移。

```bash
python run_pipeline.py --models-only
python run_figures.py --skip-training
```

| 模型 | D1 Test RMSE | 论文参考 |
|------|-------------|----------|
| XGBoost | 1.09% | 1.1% |
| SVR | 1.02% | 1.1% |

迁移 TL2：D2 4.14%，D3 5.41%（论文 TL2：1.7% / 1.6%）。

结果见 `output/results.json`。

---

## 与作者原始代码的关系

| 内容 | 作者提供 | 本仓库 |
|------|----------|--------|
| 原始循环 CSV | ✅ | 直接使用 |
| 弛豫电压段截取 | ✅ 部分脚本 | ✅ |
| 统计特征 + SVR/XGBoost | ❌ | ✅ `battery_pipeline/` |
| 迁移学习 TL2 | ❌ | ✅ |
| MS-CNN MAE + 门控融合 | ❌ | ✅ `research_mae/` |
| Quantile TCN RUL 预测 | ❌ | ✅ `research_rul/` |

---

## 注意事项

1. **内存**：特征提取流式处理，正常运行 < 2 GB。
2. **PyTorch**：默认 `--device cpu`；GPU 驱动过旧时请保持 CPU 模式。
3. **随机性**：Strategy D 划分固定 `random_state=42`；Fusion 集成 seeds `42, 43, 44`。
4. **数据重建**：序列维度变更或 CC 过滤更新后，需 `--rebuild-data` 重建缓存。

---

## 参考文献

Zhu, J., Wang, Y., Huang, Y. et al. Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation. *Nat Commun* **13**, 2261 (2022). https://doi.org/10.1038/s41467-022-29837-w
