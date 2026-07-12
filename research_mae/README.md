# 研究内容一：弛豫电压掩码自编码器 + 通道注意力融合

基于 Zhu et al. (2022) 公开数据集，实现：

1. 充电结束后弛豫电压序列截取与**统一固定维度**重采样  
2. **时序卷积掩码自编码器（MAE）** 提取极化衰退隐变量  
3. **通道注意力** 融合弛豫隐变量与恒流（CC）充电时间  
4. 生成 Fig 1–5

## 目录结构

```
research_mae/
  data_extract.py    # 弛豫 ΔV 序列 + CC 充电时间提取
  features.py        # CC 相对老化特征
  models.py          # TemporalMaskedAE + GatedChannelFusion
  train.py           # 训练与 checkpoint（含 history）
  transfer_fusion.py # D2 TL 微调（冻结门控 + 重训 head）
  plot_training.py   # 训练曲线绘图
  evaluate.py        # Strategy D 容量回归评估
  figures.py         # Fig 1–6 绘图
  run_all.py         # 一键运行
  cache/             # 提取后的 npz 缓存
  checkpoints/       # 模型权重
  figures/           # 输出图片
```

## 运行

```bash
conda activate battery-capacity
cd /path/to/data-driven-capacity-estimation-from-voltage-relaxation

# 完整流程（提取 → 训练 MAE → 训练注意力融合 → 出图）
PYTHONUNBUFFERED=1 python research_mae/run_all.py --device cpu

# 仅重建数据缓存
python research_mae/run_all.py --rebuild-data --skip-train

# 仅重训 Fusion（复用 MAE checkpoint）
python research_mae/run_all.py --skip-mae --device cpu

# 使用已有 checkpoint 只出图 + 评估
python research_mae/run_all.py --skip-train --device cpu
```

## 数据处理说明

| 数据集 | 弛豫时长 | 固定序列长度 | ΔV 定义 |
|--------|----------|--------------|---------|
| Dataset 1/2 | 30 min | 30 点 | V(t) − V(t₀) |
| Dataset 3 | 60 min | 60 点 | V(t) − V(t₀) |

- 在 `[0, T_relax]` 上均匀重采样，实现**统一固定维度**  
- CC 充电时间：每圈恒流充电段（`control/V/mA ≈ control/mA` 且 I>0）的持续时间（秒）

## 模型说明

### 掩码自编码器（MAE）

- 输入：`(batch, 1, seq_len)` 的 ΔV 序列  
- Encoder（1D-CNN）→ 隐向量 `z`（32 维）  
- Decoder（1D-CNN）从 `z` 重构完整序列  
- 训练完成后：`encode(ΔV)` 即为**极化衰退核心特征**

### 通道注意力融合（Fig 5）

- 两路输入：弛豫隐向量 + CC 充电时间（标量嵌入）  
- Softmax 注意力权重 → 加权融合，用于容量相关监督  
- Fig 5 绘制两路权重随循环圈数的变化

## 输出图表

| 文件 | 内容 |
|------|------|
| `fig1_relaxation_delta_v.png` | 第 10/300/600 圈 ΔV–时间曲线 |
| `fig2_cc_time_dataset*.png` | CC 充电时间随循环退化 |
| `fig3_mae_recon_dataset*.png` | 初/中/末期 MAE 掩码重构对比 |
| `fig4_latent_manifold_single_cell.png` | 单电芯隐向量 t-SNE 老化轨迹 |
| `fig4_latent_manifold_all.png` | 全数据集隐向量流形 |
| `fig5_attention_weights.png` | 通道注意力权重随老化演变 |
| `fig5_attention_by_condition.png` | 按 C-rate 工况分组的注意力 |
| `fig6_capacity_prediction.png` | Strategy D 测试集容量预测散点 |
| `train_mae_short.png` / `train_mae_long.png` | MAE 训练 loss / smooth / lr |
| `train_fusion_ds1/2/3.png` | 各数据集 Fusion 训练曲线 |
| `train_overview_val_loss.png` | MAE + Fusion 验证 loss 总览 |
| `fig7_transfer_comparison.png` | D1 模型零样本迁移 D2/D3 RMSE 对比 |

## 定量结果（Strategy D 留出电芯）

| 方法 | Test RMSE% | R² |
|------|------------|-----|
| **D1 集成 (3 seeds)** | **0.57%** | 0.99 |
| D1 单模型 | 0.75% | — |
| D2 零样本 | 1.16% | — |
| D2 原生留出 | **0.36%** | 0.99 |
| D3 原生留出 | 0.94% | 0.98 |

详见 `research_mae/output/metrics.json`。

详见 [FIGURES.md](./FIGURES.md)（**每张图的含义与读图指南**）、[IMPLEMENTATION.md](./IMPLEMENTATION.md)（实现与 Debug 记录）、[PAPER_METHODS_RESULTS.md](./PAPER_METHODS_RESULTS.md)（**论文方法 + 实验结果**）。

## 依赖

- PyTorch（`pip install torch`）
- 其余见项目根目录 `requirements.txt`
