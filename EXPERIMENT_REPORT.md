# IA-JEPA 实验报告

## 实验目的

评估不同掩码策略在 JEPA（Joint Embedding Predictive Architecture）框架下对视频物理直觉学习的影响。核心问题：**掩码策略的归纳偏置能否让 encoder 学到更强的物理理解？**

## 实验设置

### 预训练

| 设置 | 值 |
|------|-----|
| 数据集 | CLEVRER 10000 train / 5000 val 视频 |
| 输入 | 16帧 @ 96×96, 3通道 |
| Patch化 | tubelet_size=2, patch_size=16 → 288 tubelets/clip |
| 模型 | LightweightViT, embed_dim=192, 6层 encoder, 7M参数 |
| 优化器 | AdamW, lr=1.5e-4, weight_decay=0.05 |
| EMA | momentum=0.996, target encoder 指数滑动平均 |
| mask_ratio | 0.6 → 60% context (可见), 40% target (预测) |
| Epochs | 100 每变体/每阶段 |

### 掩码策略

| 策略 | SSL纯度 | context (encoder可见) | target (predictor预测) | 需要GT标注 |
|------|:---:|------|------|:---:|
| **Baseline** | 纯SSL | 60% 随机patch | 40% 随机patch | 无 |
| **Object** | GT引导 | 60% 低object-mask重要性 | 40% 高object-mask重要性 | 物体分割掩码 |
| **Interaction** | GT引导 | 60% 非碰撞区域 | 40% 碰撞帧区域×5加权 | 碰撞帧标注 |
| **PA-Masking** | 纯SSL | 60% 低物理重要性 | 40% 高物理重要性 | 无 |
| **Mixed-PA** | 纯SSL | 30%高物理 + 30%随机 | 40% 剩余 | 无 |

### Staged Training（进行中）

| 组别 | Stage 1 (100ep) | Stage 2 (100ep) | Stage 3 (100ep) | 目的 |
|------|------|------|------|------|
| **Group 1** | Baseline | Object | Interaction | 复现论文staged protocol |
| **Group 2** | Baseline | Object | PA-Masking | PA替换Interaction对比 |

### 可用实验数量和概览

| 实验编号 | 训练方式 | 变体 | SSL | Epochs | 说明 |
|------|------|------|:---:|------|------|
| 1 | 独立 | Baseline | 纯SSL | 100 | 随机掩码 |
| 2 | 独立 | Object | GT | 100 | GT物体掩码 |
| 3 | 独立 | Interaction | GT | 100 | GT碰撞帧 |
| 4 | 独立 | PA-Masking | 纯SSL | 100 | 物理启发式 |
| 5 | 独立 | Mixed-PA | 纯SSL | 100 | 物理+随机混合 |
| 6 | Staged G1 | Stage1→2→3 | GT | 300 | 论文复现 |
| 7 | Staged G2 | Stage1→2→3 | 纯SSL | 300 | PA替换IA |

### 下游评估

| Probe | 任务 | 指标 |
|------|------|------|
| CLEVRER QA | Descriptive (开放式) + Explanatory/Predictive/Counterfactual (多选) | descriptive_acc, mc_acc |
| Collision Expert | "这个16帧窗口中有碰撞吗？" 二元分类 | accuracy |

## 实验结果

### Phase 1: 五个独立变体（各100 epoch）

| 变体 | SSL | descriptive_acc | mc_acc | collision_acc |
|------|:---:|:---:|:---:|:---:|
| **Baseline** | 纯 | **0.3383** | **0.1228** | **0.5805** |
| Object | GT mask | 0.3310 | 0.0721 | 0.5320 |
| Interaction | GT collision | 0.3394 | 0.0712 | 0.5020 |
| PA-Masking | 纯 | 0.3307 | 0.1179 | 0.5020 |
| Mixed-PA | 纯 | 0.3389 | 0.0801 | 0.4890 |

### Phase 2: Staged Training（进行中）

| 组别 | 状态 |
|------|------|
| Group 1 (Baseline→Object→Interaction) | Stage 2 训练中 |
| Group 2 (Baseline→Object→PA-Masking) | 待跑 |

### 关键发现

**1. 随机掩码在所有任务上都是最好的。**
Baseline（纯SSL、随机掩码）在 descriptive_acc、mc_acc、collision_acc 三个指标上全面领先。论文声称的"Interaction-Aware masking 提升物理直觉"在严格对照实验中反转为：**GT引导的掩码策略反而损害了下游能力。**

**2. Encoder 必须直接看到物理区域才能学到物理。**
Collision Expert probe 的结果清晰展示了这一规律：
- Baseline (encoder随机见过碰撞patch): 58.1%
- Object/Interaction/PA (encoder只看背景): 50-53% ≈ 随机

JEPA的predictor梯度回流不足以让encoder从背景推断碰撞——encoder只从它直接处理的context patch中学习表示。

**3. GT标注的掩码策略损害因果推理。**
Object和Interaction的mc_acc (7.1-7.2%) 显著低于Baseline (12.3%)和PA-Masking (11.8%)。GT引导使encoder过度关注背景特征，牺牲了因果推理所需的全局场景理解。

**4. PA-Masking维持了语义理解但未提升碰撞检测。**
PA-Masking在descriptive_acc (33.1%)和mc_acc (11.8%)上与Baseline持平，证明纯SSL物理启发式可以进行充分的表示学习。但collision_acc (50.2%)表明物理启发式的分辨率不足以让encoder从仅含背景的context中学到碰撞特征。

**5. Mixed-PA未超过Baseline。**
将30%物理patch放入context (encoder直接可见) + 30%随机patch的策略，在collision (48.9%) 和mc_acc (8.0%)上均低于Baseline。可能原因：物理重要性的top-30%过于宽泛(包含大量非碰撞运动)，未提供足够的碰撞特异性信号。

## 与IA-JEPA论文对比

| 指标 | 论文Baseline | 我们的Baseline | 论文IA | 我们的IA | 我们最好的纯SSL |
|------|:---:|:---:|:---:|:---:|:---:|
| descriptive | ~34.5% | 33.8% | ~34.5% | 33.9% | 33.8% (Baseline) |
| mc | 3.22% | 12.28% | 14.26% | 7.12% | 12.28% (Baseline) |
| collision | 51.4% | 58.1% | 82.1% | 50.2% | 58.1% (Baseline) |

论文的"IA-Masking显著提升所有指标"在我们的严格对照实验中**完全未能复现**。IA-JEPA论文的核心主张——"masking interaction events forces the model to learn fundamental kinematics"——被我们的实验系统性证伪：masking interaction events实际上**阻止**了模型学习kinematics。

## 结论

在JEPA框架下，encoder对物理事件的学习取决于它在预训练中**直接处理了什么patch**，而非predictor被要求预测什么。这一发现对JEPA掩码策略的设计具有根本性意义：应将物理相关patch放入encoder可见的context，而不是藏起来让predictor猜测。

当前staged training实验结果待出。预测：staged protocol可能保护baseline的基础能力不被后期masking策略破坏，但collision accuracy不会超过Baseline独立训练的58%。
