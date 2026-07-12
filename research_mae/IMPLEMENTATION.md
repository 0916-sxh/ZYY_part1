# 研究内容一：实现与 Debug 说明

本文档记录 `research_mae/` 模块的**总体思路、实现步骤、Debug 修改**，便于复现。

> 论文撰写用方法 + 实验结果见 **[PAPER_METHODS_RESULTS.md](./PAPER_METHODS_RESULTS.md)**。

---

## 一、总体思路（未改）

```
原始 CSV
  → 截取满充后弛豫电压 ΔV 序列（固定维度重采样）
  → 提取 CC 恒流充电时间
  → 时序卷积掩码自编码器（MAE）学习隐变量
  → 通道注意力融合 [弛豫隐变量, CC时间]
  → 输出 Fig 1–5
```

核心假设不变：**弛豫电压序列含极化衰退信息**；MAE 通过掩码重构迫使网络学习序列结构；隐向量用于老化表征。

---

## 二、数据处理

### 2.1 弛豫电压截取（Debug 重点）

**问题（初版）**：`_find_relax_start` 找到的是第一个 `control=0` 段，可能是充电前静置，且 Fig 1 把放电前所有零电流段都画进去，出现 7000 s 平台+陡降。

**修复**：新增 `find_post_charge_relaxation()`：

1. 先定位放电起点：`current < -50 mA`
2. 在放电前找满足以下条件的**最后一段**零电流区间：
   - `|control| ≈ 0`
   - `|current| < 5 mA`
   - `voltage > 4.0 V`
3. 从该段起点截取固定时长：
   - Dataset 1/2：**30 min** → 重采样 **30 点**
   - Dataset 3：**60 min** → 重采样 **60 点**
4. 窗口终点再**剔除放电切换点**：`|I|>0.5 mA` 或单步 ΔV>15 mV 的尾部点
5. 计算 **ΔV = V(t) − V(t₀)**

实现文件：`research_mae/data_extract.py`

### 2.2 CC 充电时间

取**第一段**连续 CC 区间（`I>50 mA`、`control/V/mA≈control/mA`、`V<4.19 V`），即从较低电压恒流充至 CV 门槛的时长。数值约 3000–6000 s（0.5C 充 3.5 Ah 量级），随老化单调下降，Fig 2 趋势合理。

### 2.3 固定维度与归一化

- 在 `[0, T_relax]` 上均匀插值到固定长度
- 训练用 **z-score 归一化**（全局 μ、σ 存入 cache）
- 原始 mV 级 ΔV 保存在 `delta_v_raw` 供绘图还原

---

## 三、掩码自编码器（MAE）

### 结构

| 组件 | 配置 |
|------|------|
| Encoder | Conv1d 1→32→64→128 + GAP + Linear → **32 维**隐向量 |
| Decoder | Linear → Conv1d 128→64→32→16→1 |
| 掩码比例 | 30% 时间步随机置零 |
| 损失 | 掩码位置 MSE + **0.05×平滑正则**（减少重构锯齿） |

### 训练

- Dataset 1+2 合并训练 `mae_short`（seq=30）
- Dataset 3 单独训练 `mae_long`（seq=60）
- 默认 **40 epoch**，Adam lr=1e-3

实现文件：`research_mae/models.py`、`research_mae/train.py`

---

## 四、通道注意力融合（Fig 5 Debug）

### 问题（初版）

注意力权重塌缩：弛豫=1.0，CC=0.0，无动态变化。

### 修复

1. CC 时间改为 **z-score 标准化**（保留跨循环变化）
2. 双路分别经 `Linear` / `MLP` 投影到同维度
3. 损失函数增加：
   - **熵正则** `−λ·H(w)`（λ=0.15，鼓励权重分散）
   - **平衡项** `0.05·Σ(mean(w)−0.5)²`（弱约束，避免权重完全冻结在 0.5）
4. 训练 epoch 增至 **80**

实现文件：`research_mae/models.py` → `ChannelAttentionFusion`

---

## 五、图表生成

| 图 | 文件 | 说明 |
|----|------|------|
| Fig 1 | `fig1_relaxation_delta_v.png` | 选循环数≥600 的电芯，画 10/300/600 圈 ΔV–时间 |
| Fig 2 | `fig2_cc_time_dataset*.png` | CC 时间随圈数变化 |
| Fig 3 | `fig3_mae_recon_dataset*.png` | 初/中/末期：原始、30%掩码、重构 |
| Fig 4 | `fig4_latent_manifold_single_cell.png` | **单电芯** t-SNE 老化轨迹 |
| Fig 4 | `fig4_latent_manifold_all.png` | 全数据集流形（对比用） |
| Fig 5 | `fig5_attention_weights.png` | 两路注意力权重随圈数 |

实现文件：`research_mae/figures.py`

---

## 六、运行方式

```bash
conda activate battery-capacity
cd /path/to/data-driven-capacity-estimation-from-voltage-relaxation

# 完整流程：重建缓存 + 训练 + 出图
PYTHONUNBUFFERED=1 python research_mae/run_all.py --rebuild-data --device cpu

# 仅重新出图（已有 checkpoint）
python research_mae/run_all.py --skip-train --device cpu
```

输出目录：
- 数据缓存：`research_mae/cache/`
- 模型权重：`research_mae/checkpoints/`
- 图片：`research_mae/figures/`

---

## 七、目录结构

```
research_mae/
├── IMPLEMENTATION.md      # 本文档
├── README.md              # 简要说明
├── data_extract.py        # 数据截取 + 缓存
├── models.py              # MAE + 注意力
├── train.py               # 训练逻辑
├── figures.py             # Fig 1–5
├── training_log.py      # 训练历史 dataclass
├── plot_training.py     # 训练曲线绘图
├── evaluate.py          # Strategy D 容量回归评估
├── run_all.py           # 入口
├── cache/               # npz 缓存
├── checkpoints/         # 模型 pt
├── output/              # metrics.json
└── figures/             # 输出 png
```

---

## 八、Debug 前后对比（预期改善）

| 项目 | Debug 前 | Debug 后 |
|------|----------|----------|
| Fig 1 曲线形态 | 7000 s 异常平台 | 30 min 内平滑弛豫下降 |
| Fig 1 圈数 | 仅 Cycle 10 | 10/300/600 三圈 |
| CC 时间量级 | 取最长段 ~10000 s | 首段 CC（3V→4.19V）约 4000–6000 s，趋势随老化下降 |
| Fig 3 重构 | 锯齿严重 | 平滑正则后更贴近原始 |
| Fig 4 | 多电芯混杂 | 增加单电芯清晰轨迹 |
| Fig 5 注意力 | 恒为 1/0 | 0.45–0.55 动态变化，晚期弛豫权重升高 |
| 容量回归 RMSE | 未评估 | Strategy D 测试集 **0.55%**（优于论文 SVR ~1.02%） |

---

## 九、继续优化（第二轮）

1. **训练**：MAE/Fusion 增加电芯级验证、AdamW + Cosine LR、早停；Fusion 仅在 Strategy D 训练电芯上拟合  
2. **评估**：新增 `evaluate.py`，Strategy D 留出电芯上报告 RMSE%/R²  
3. **Fig 6**：容量预测散点图；**Fig 5b**：按 C-rate 分组注意力  
4. **训练曲线（第三轮）**：每 epoch 记录 loss / lr / 注意力 / 熵，输出 `train_mae_*.png`、`train_fusion_*.png`、`train_overview_val_loss.png`  
5. **梯度裁剪**：MAE 与 Fusion 均 `clip_grad_norm=1.0`

训练日志 JSON：`research_mae/output/history_*.json`

---

## 十、继续优化（第五轮 — 门控融合）

1. **GatedChannelFusion**：独立 sigmoid 门控替代 softmax，避免 0.5/0.5 塌缩  
2. **CC 双特征**：`log(CC/CC₀)` 相对老化 + 全局 z-score  
3. **D3 专用 Fusion**：long MAE 隐向量 + 15% 电芯留出训练  
4. **D1 测试 RMSE 0.84%**，已优于论文 SVR 1.02%

---

## 十一、继续优化（第六轮）

1. **SmoothL1 损失** + 更低 lr/weight_decay，D1 测试 RMSE **0.55%**  
2. **D2 原生 Fusion**（`fusion_ds2`）留出 RMSE **0.33%**  
3. **D2 TL 微调**：冻结 D1 门控，稀疏样本重训 head → 1.51%（优于零样本 1.92%）  
4. **D3 Fusion** 留出 RMSE **0.47%**

---

## 十二、继续优化（第七轮 — 高性价比）

1. **隐向量 z-score**（训练集 μ/σ）— 稳定 Fusion 输入  
2. **D1 三 seed 集成**（42/43/44）— 预测取均值，默认 `--fusion-seeds 42,43,44`  
3. **自动汇总** `output/RESULTS.md`  
4. 修复 TL 微调中隐向量未归一化的问题  

---

## 十三、依赖

- Python 3.10+
- PyTorch、numpy、pandas、scikit-learn、matplotlib

见项目根目录 `requirements.txt`。
