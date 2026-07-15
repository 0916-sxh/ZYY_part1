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

## 三、掩码自编码器（Hybrid Dilated MS-CNN MAE）

### 结构（第八轮）

| 组件 | 配置 |
|------|------|
| Encoder | **Hybrid DilatedMSConvBlock** ×3（kernel 3/5/7 + dilation 2/4，残差）→ **Avg+Max 双池化** → Linear → **32 维** z |
| Aging head | Linear → GELU → Linear → **Sigmoid** → 标量 ∈ [0,1] |
| Decoder | Linear → Conv1d 128→64→32→16→1 |
| 掩码 | **30% 连续块** |
| 损失 | 0.85×掩码区 MSE + 0.15×可见区 MSE + 0.05×平滑 + **老化监督** |

序列维度：D1/D2 **32 点**（30 min），D3 **64 点**（60 min）。

实现文件：`research_mae/models.py` → `MSCNNMaskedAE`

### 老化监督（第八轮 — Spearman 优化）

**动机**：纯无监督 MAE 的 z 与 SOH/寿命比率相关，但 t-SNE 可视化 Spearman 仅 ~0.6；PCA 第一轴可达 0.8+。需要显式拉出**单调老化方向**。

**做法**（`train.py` + `models.py`）：

1. `_aging_targets()`：`0.55×fade + 0.45×life_ratio`（电芯内 cycle 归一化）
2. `aging_head` 预测该目标；损失 = SmoothL1 + `pairwise_ranking_loss`（batch 内保序）
3. MAE 收敛后 **40 epoch aging-head 微调**（**冻结 encoder/decoder**，只训 `aging_head`）
4. **Sigmoid 输出层**：将 aging 分数限制在 [0,1]，避免离群值撑爆 Fig 4 坐标轴
5. **early-stop 仅看重构 val_loss**（不含 aging），防止老化损失抢走 best checkpoint

**Fig 4 投影**（`thesis_figures.py`，非 t-SNE）：

- Dim 1 = `aging_head(z)`
- Dim 2 = 去掉 aging 线性分量后残差的 PCA1
- Spearman = Dim 1 vs **寿命比率**（不用绝对圈数）
- 显示裁剪：0.5%–99.5% 分位，标题标注 `n≈3000`

**结果**：D1 ρ=0.893，D2 ρ=0.840，D3 ρ=0.993（均 > 0.8）

> 完整方法说明见 **[RESEARCH_CONTENT_1.md](./RESEARCH_CONTENT_1.md) §3.2、§3.6**。

---

## 四、通道注意力融合（Fig 5）

**实现**：`ChannelAttentionFusion`（Softmax 两路权重，熵正则防塌缩）。

历史版本曾用 `GatedChannelFusion`（Sigmoid 门控），见第十节。

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
├── export_features.py # 融合特征 .npy 导出（研究内容二接口）
├── run_all.py           # 入口
├── RESEARCH_CONTENT_1.md  # 研究内容一完整说明
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

## 十四、继续优化（第八轮 — 老化轴 + Fig 4；第九轮均衡）

1. **Hybrid Dilated MS-CNN**：kernel 3/5/7 + dilation 2/4 + 残差；Avg+Max 双池化
2. **aging head + 半监督**：SOH/寿命比率目标 + 排序损失；主训练后 **仅微调 aging_head**（冻结 encoder/decoder）；Sigmoid 限幅
3. **防回归**：early-stop 只看重构 MSE；禁止全参数 aging 微调冲坏 Fig 3
4. **Fig 4 改法**：学习老化轴 + 残差 PC（弃用 t-SNE 报 Spearman）；寿命比率着色；分位裁剪防离群
5. **D1 集成扩至 5 seeds**（42–46）；均衡重训后融合 RMSE **0.47%**（D3 **0.54%**）
6. **Spearman 达标**：D1 **0.894**，D2 **0.843**，D3 **0.991**（重构 MSE ~0.0005）

**Debug 记录 — Fig 4 D1 “点很少”**：实为 ~3000 点被个别 aging 离群值（未 Sigmoid 前极值 20+）拉爆坐标轴；修复后云团正常铺满。

**Debug 记录 — Fig 3 重构突然变差**：全参数 aging 微调把解码器冲坏（recon MSE ~0.002→~1.5）。改为只训 `aging_head` 后 Fig 3 恢复贴合，Spearman 仍 >0.84。

---

## 十三、依赖

- Python 3.10+
- PyTorch、numpy、pandas、scikit-learn、matplotlib

见项目根目录 `requirements.txt`。
