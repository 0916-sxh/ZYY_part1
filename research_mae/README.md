# 研究内容一：弛豫电压 MS-CNN MAE + 通道注意力融合

基于 Zhu et al. (2022) 公开数据集，实现开题报告**研究内容一**：

1. 充电结束后弛豫电压序列截取与**固定 32/64 维**重采样  
2. **多尺度卷积（MS-CNN）掩码自编码器** 无监督提取极化衰退隐变量  
3. **Softmax 通道注意力** 融合弛豫隐向量与 CC 充电时间  
4. 导出 `.npy` 融合特征（供研究内容二/三使用）  
5. 生成 Fig 1–7

**完整说明** → [RESEARCH_CONTENT_1.md](./RESEARCH_CONTENT_1.md)

## 快速运行

```bash
conda activate battery-capacity
cd /path/to/data-driven-capacity-estimation-from-voltage-relaxation

PYTHONUNBUFFERED=1 python research_mae/run_all.py --rebuild-data --device cpu
```

## 核心文件

| 文件 | 作用 |
|------|------|
| `models.py` | `MSCNNMaskedAE` + `ChannelAttentionFusion` |
| `export_features.py` | 导出 `features/dataset_*_fused.npy` |
| `run_all.py` | 训练 + 评估 + 导出 + 出图 |

## 文档索引

- [RESEARCH_CONTENT_1.md](./RESEARCH_CONTENT_1.md) — 方法、运行、开题对齐  
- [IMPLEMENTATION.md](./IMPLEMENTATION.md) — Debug 与迭代记录  
- [FIGURES.md](./FIGURES.md) — 图表读图指南  
- [PAPER_METHODS_RESULTS.md](./PAPER_METHODS_RESULTS.md) — 论文方法与结果  
