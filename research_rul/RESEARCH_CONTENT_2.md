# 研究内容二：物理约束 + 不确定性量化的 RUL 预测

基于研究内容一导出的融合特征 `research_mae/features/dataset_*_fused.npy`，采用 **Quantile TCN** 预测剩余寿命（RUL），并用 Pinball Loss + 单调性物理惩罚实现不确定性量化。

## 方法

| 模块 | 说明 |
|------|------|
| `rul_labels.py` | RUL = N_EOL − cycle；EOL = 80% 标称容量；删失电芯剔除 |
| `quantile_tcn.py` | 因果 TCN + 5%/50%/95% 分位数输出 |
| `losses.py` | Pinball Loss + λ×单调惩罚 |
| `dataset.py` | 历史融合特征序列 → RUL 样本 |
| `train.py` | Strategy D 划分、消融实验 |
| `figures.py` | Fig 6–9 |

## 运行

```bash
conda activate battery-capacity

# 仅研究内容二（需先完成 research_mae 特征导出）
python research_rul/run_all.py --device cpu

# 研究内容一 + 二 一键运行
python run_research.py --rebuild-data --device cpu
```

## 图表

| 图 | 文件 | 内容 |
|----|------|------|
| Fig 6 | `fig6_monotonic_penalty.png` | 有/无单调惩罚 RUL 预测对比 |
| Fig 7 | `fig7_ablation.png` | 四特征消融 RMSE/MAE |
| Fig 8 | `fig8_rul_confidence.png` | RUL 中位数 + 90% 置信区间 |
| Fig 9 | `fig9_transfer_dataset*.png` | D1→D2/D3 零样本迁移 |

## 消融特征模式

1. **latent** — 仅弛豫隐向量  
2. **cc** — 仅 CC 宏观特征  
3. **concat** — 简单拼接  
4. **fused** — 门控跨尺度融合（研究内容一输出）

## 评估指标

- RMSE / MAE（圈数）
- PICP（90% 区间覆盖率，目标 ≈ 0.90）
- PINAW（归一化区间宽度）

## 当前结果（Strategy D 测试集）

| 指标 | 数值 |
|------|------|
| RMSE | **24.3** 圈 |
| MAE | 13.7 圈 |
| PICP (90%) | 0.49 |
| PINAW | 0.13 |

### 消融（Fig 7）

| 特征输入 | RMSE | MAE |
|---------|------|-----|
| 仅弛豫 latent | 37.0 | 20.7 |
| 仅 CC | 29.0 | 15.4 |
| 拼接 concat | 32.4 | 18.2 |
| **融合 fused** | **24.1** | **13.3** |

RUL 标签：80% EOL，56/66 电芯有效，10 颗删失剔除。

输出：`research_rul/output/`、`research_rul/figures/`（Fig 6–9）
