# 研究内容一：弛豫电压 Hybrid Dilated MS-CNN MAE + 门控融合

基于 Zhu et al. (2022) 公开数据集，实现开题报告**研究内容一**：

1. 充电结束后弛豫电压序列截取与**固定 32/64 维**重采样  
2. **Hybrid Dilated MS-CNN 掩码自编码器** + **aging head** 老化轴监督  
3. **门控通道融合** 融合弛豫隐向量与 CC 充电时间  
4. 导出 `.npy` 融合特征（供研究内容二/三使用）  
5. 生成论文规格 **Fig 1–5、Fig 10**（`thesis_figures.py`）

**完整说明** → [RESEARCH_CONTENT_1.md](./RESEARCH_CONTENT_1.md)

## 快速运行

```bash
conda activate battery-capacity
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd /path/to/data-driven-capacity-estimation-from-voltage-relaxation

# 完整训练 + 导出 + 出图（GPU）
python research_mae/run_all.py --device cuda --fusion-seeds 42,43,44,45,46

# 仅重画 Fig 1–5、Fig 10
python research_mae/thesis_figures.py --device cuda
```

## 核心文件

| 文件 | 作用 |
|------|------|
| `models.py` | `MSCNNMaskedAE`（Hybrid Dilated MS-CNN + aging head）+ `GatedChannelFusion` |
| `train.py` | MAE 老化监督训练 + Fusion 训练 |
| `thesis_figures.py` | 论文规格 Fig 1–5、Fig 10（Fig 4 = 老化轴流形） |
| `export_features.py` | 导出 `features/dataset_*_fused.npy` |
| `run_all.py` | 训练 + 评估 + 导出 + 出图 |

## 关键指标（当前）

| 指标 | D1 | D2 | D3 |
|------|----|----|-----|
| Fusion RMSE% | **0.43** | **0.23** | **0.60** |
| Fig 4 Spearman ρ | **0.893** | **0.840** | **0.993** |

## 文档索引

- [RESEARCH_CONTENT_1.md](./RESEARCH_CONTENT_1.md) — 方法、运行、开题对齐  
- [IMPLEMENTATION.md](./IMPLEMENTATION.md) — Debug 与迭代记录（含第八轮老化轴）  
- [FIGURES.md](./FIGURES.md) — 图表读图指南  
- [PAPER_METHODS_RESULTS.md](./PAPER_METHODS_RESULTS.md) — 论文方法与结果  
