# Idea-A — Apo→Holo 口袋最小变形流（IMF）

## 范式覆盖
- Conditional Flow Matching（生成）  
- 几何可行域投影（**硬约束 / 投影层**）

---

### 1. 核心主张
提出 **Invariant-Manifold Flow (IMF)** —— 用条件流匹配学习 apo→holo 口袋形变（唯一训练目标），并用几何可行域投影层（硬约束、非损失实现）在前向过程中强制姿势合法性，从而在 OOD（CrossDocked-LFO / skeleton OOD）下稳定提升 **Top-1/Top-5 RMSD ≤ 2Å 与 EF1%**。

---

### 2. 动机与痛点（升级版）
- **观测现象（指标化）**：在 CrossDocked、PoseBusters 等基准下，生成式方法在“高曲率 / 狭窄口袋”区域出现：  
  1) 高穿插率（碰撞/重叠）；2) 键长/键角异常；3) Top-k RMSD 波动；4) EF1% 下滑。

- **逐文献局限（示例）**：  
  - *FlowDock (2024)*：apo→holo + 评分，但对穿插仅靠事后筛除，**无法保证几何合法性**。  
  - *DynamicBind (2024)*：等变扩散建模柔性，采样链在狭窄腔体**震荡严重**。  
  - *FABind 系列*：端到端口袋预测+对接，**形变建模弱**，OOD 下难保合法性。  
  - *FlexSBDD / NeurIPS 2024*：强调柔性，但仍主要靠 **soft-loss / 打分筛选**。

**结论**：现有方法多为“生成 + 打分/加损失”，**缺少前向投影到几何可行域的机制**；因此提出 IMF 用投影替代堆损失。

---

### 3. 假设与证伪
- **正向假设**：存在 L-Lipschitz 的形变映射 \( \Phi_\theta \)，且投影算子 \( \Pi_{\mathcal C} \) 为非扩张映射：  
  $$
  \lVert \Pi_{\mathcal C}(x)-\Pi_{\mathcal C}(y)\rVert \le \lVert x-y\rVert .
  $$
  则一次“生成 + 投影”能显著降低穿插率并收缩 RMSD 偏差。
- **反向假设**：若 \( \Pi_{\mathcal C} \) 不是收缩或可行域定义过松，性能不提升。
- **替代假设**：用额外 soft-loss 代替硬投影，但需更高采样预算才能达到相同合法率。

**最小证伪实验**：固定采样步数与种子，对比三组（A：生成+投影；B：生成+soft-loss；C：baseline），指标：穿插率、Top-1 RMSD、EF1%（配对检验）。

---

### 4. 指标与阈值
- **主指标**：Top-1/Top-5 RMSD ≤ 2Å（目标提升 ≥ 2–3%）；EF1%（目标 ≥ +2–3%）；PoseBusters 合法率（碰撞/穿插率显著下降）。
- **数据集**：PDBBind（ID）、CrossDocked-LFO（跨骨架/跨家族 OOD）、PoseBusters（几何合法性）、DEKOIS2.0 / DUD-E（虚筛 EF1%）。

---

### 5. 验证协议
- 三个独立随机种子（≥ 3 seeds），95% 置信区间，统计功效设定 0.8（以 EF1% 与 Top-1 RMSD 差异估样本量）。  
- 报告：Top-1/5 RMSD 分布、EF1%、穿插率、键长/键角异常率；配对 t 或 Wilcoxon，并做多重比较校正（如 Holm–Bonferroni）。

---

### 6. 文献对比与创新性（2024–2025 近作）
- *FlowDock (2024)*：解决 apo→holo 生成与置信预测；**我方**：在其基础上引入**投影层**保证几何合法性——不是再加一个 loss。  
- *DynamicBind (2024)*：解决柔性场景生成；**我方**：替代多步扩散采样在窄腔中震荡，通过**单次投影**降低穿插。  
- *FABind*：解决盲对接一体化速度与端到端训练；**我方**：把“形变 + 合法化”分成**可证的二段范式**，且投影为硬约束（非损失）。

**创新要点（不可替代）**：将 **条件流匹配** 与 **可微/可替代的几何投影层** 作为范式单元（投影是可插拔算子，而非新 loss），并给出**可证伪评测**。

---

### 7. 理论结果（完整逻辑）
- **假设**：\( \Phi_\theta \) 为 L-Lipschitz；\( \Pi_{\mathcal C} \) 为非扩张；口袋 SDF 在待测区间有下界 \( \text{SDF}(x) \ge \delta > 0 \) 。
- **结论一（误差收缩）**：由非扩张性与 Banach 不动点类论证可得 RMSD 偏差收缩：  
  $$
  \mathbb{E}\!\left[\operatorname{RMSD}\big(\Pi_{\mathcal C}(\Phi_\theta(x)), x^\star\big)\right] 
  \le \alpha(L)\, \mathbb{E}\!\left[\operatorname{RMSD}\big(\Phi_\theta(x), x^\star\big)\right],\quad \alpha(L)\!<\!1 .
  $$
- **结论二（穿插率上界）**：存在常数 \( C(\delta, L) \) 使得  
  $$
  \mathbb{P}\!\left[\text{Overlap}\!\left(\Pi_{\mathcal C}(\Phi_\theta(x))\right)\right]
  \le C(\delta, L)\cdot \mathbb{P}\!\left[\text{Overlap}\!\left(\Phi_\theta(x)\right)\right].
  $$
- **适用边界（失效）**：当 apo→holo 为非单连通或开闭位移超阈（如口袋开闭 \(> 6\,\text{\AA}\)）时，上界可能失效。

---

### 8. 复现实验设计（MFE、Full / Lite）
- **Full 方案**：2×A100-80GB 或 4×A100-40GB，CrossDocked 全量 + PoseBusters，预算 ≤ 180 GPUh。  
- **Lite 方案**：1×4090 或 ≤ 3×4080，PDBBind-Refined 子集 + PoseBusters 子集，预算 ≤ 40–60 GPUh。  
- **实现要点（2 天可起跑）**：在现有 CFM/flow 框架中实现 \( \Pi_{\mathcal C} \)（SDF-based projection + local bond–angle fixer），训练仅最小改动（保持原生成 loss）。

---

### 9. 负结果利用
若无提升，构建“**失败口袋集合**”（按口袋开闭幅度、SDF 梯度下界、狭窄度），并作为新的 OOD 难例评测基准公开。

---

### 10. 跨学科价值
把“生成模型 + 可行域投影”范式带入药物设计流水线，可用于蛋白–蛋白对接、RNA–小分子对接与机器人路径规避（碰撞自由轨迹）。

---

### 11. 投稿定位与博士论文映射
- **方法/理论**：NeurIPS / ICLR / ICML  
- **应用/评测**：Bioinformatics / TKDD  
- **博士论文**：第 2 章（范式）、第 4 章（理论）、第 6 章（系统/评测）

---

### 12. 递进关系（与并行 ideas 的对比）
维度递进：在“几何合法性与 OOD 稳定性”维度显著递进；为多配体分配（Idea-B）与校准打分（Idea-C）提供稳固几何基础。

---

### 13. 实验提升保证（场景化）
在 CrossDocked-LFO / PoseBusters / skeleton OOD：穿插率显著下降，Top-1/5 RMSD ≤ 2Å 成功率提高（目标 ≥ +3%）；若失败则记录诊断矩阵（SDF 梯度/口袋开闭等）。

---

### 14. 实现伪代码（简要）
```python
# generation + projection (inference)
z = sample_noise()
P_holo_hat = flow_generator.predict(P_apo, ligand, z)

# candidate poses from holo pocket field
poses = sample_poses_from_field(P_holo_hat, K)

# projection layer: applies SDF / bond-angle / vdW fixes
poses_proj = [Pi_C(p) for p in poses]

# reject-and-resample (one-shot)
valid = [p for p in poses_proj if check_validity(p)]
if len(valid) == 0:
    poses = resample(...)
    poses_proj = [Pi_C(p) for p in poses]

return select_topk(poses_proj)
```

---

# Idea-B — 多配体竞合的 OT-拍卖内隐层（GA-Ligand）

## 范式覆盖
- 扩散/流生成（候选）  
- **Gromov–Wasserstein（GW）/ 可微拍卖**（内隐分配层）

---

### 1. 核心主张
提出 **GA-Ligand** —— 通过扩散/流生成候选配体姿势，并在**前向内部**用 GW 近似 + 可微拍卖分配作为内隐层对 \(N\) 配体进行**全局最优可行占位分配**（无冲突/协同评估），避免简单打分或贪心分配造成的冲突与不稳定。

---

### 2. 动机与痛点（升级版）
- **观测现象**：多配体/多位点场景（同一口袋或相邻位点）下，基于独立生成+打分的多配体方案常出现**互相穿插或不合理占位**，导致 EF1% 与多配体成功率大幅下降。  
- **近作局限**：
  - *GroupBind (2025)*：提出多配体生成思路，但分配多靠注意力或打分后贪心筛选，**缺少全局最优分配机制**。  
  - *FlowDock*：多配体支持弱，缺少**全局互斥/协同建模**。

**结论**：需要**机制层（内隐可微分算子）**在前向完成分配，而不是训练时加复杂损失。

---

### 3. 假设与证伪
- **正向假设**：若体素化的口袋占据与配体互作能由图结构或距离矩阵表征，则用 GW-OT 的分配代价结合拍卖近似能在多配体案例实现**冲突率与覆盖率的 Pareto 优化**。
- **反向假设**：若 OT 代价不能反映真实物理互作（例如化学相互作用非常离散），则效果不优。
- **最小证伪实验**：固定生成器，仅替换分配层（GW-auction vs greedy vs attention）并比较**冲突率、Top-k RMSD、EF1%**。

---

### 4. 指标与阈值
- **主指标**：Top-1/Top-5 RMSD ≤ 2Å（按每配体）；多配体冲突率（体素/重叠阈占比 ≤ 1%）；EF1%（总体虚筛）提升 ≥ +2–3%。
- **数据集**：PDBBind 中同靶/多配体子集、构造的多配体 CrossDocked 组合、PoseBusters 用于合法性检验。

---

### 5. 验证协议
- \(N=2/3/5\) 多配体规模曲线实验；≥ 3 seeds；95% CI；报告**时间复杂度与分配收敛速度**。

---

### 6. 文献对比与创新性
- *GroupBind (2025)*：提出多配体生成但无全局最优分配机制；**本 work**：实现 **GW-OT + 拍卖内隐层** 作为**前向可微分分配算子**，不是简单的打分或损失叠加。

**创新（不可替代）**：把“最优传输 / Gromov 匹配”做成**可插拔内隐层**，在模型前向内完成**全局分配决策**，直接输出**无冲突**多配体解。

---

### 7. 技术路线（公式与步骤）
1) 生成器输出候选集合 \( \{L_i\}_{i=1}^M \)。  
2) 定义配体内/口袋位点间的图距离矩阵 \( G^{\mathrm{lig}},\, G^{\mathrm{poc}} \)。  
3) 求解（近似） **GW-OT 分配**：  
$$
\min_{\Gamma \in U} \;\sum_{i,j,k,\ell} \big\lvert G^{\mathrm{lig}}_{ij} - G^{\mathrm{poc}}_{k\ell} \big\rvert^2\, \Gamma_{ik}\,\Gamma_{j\ell} ,
$$
其中 \( \Gamma \) 通过**拍卖近似 / Sinkhorn 变体**求解并**嵌入前向（可微）**；同时对重叠惩罚作为**投影裁剪**而非反向损失。  
4) 输出被选中的 \(N\) 个配体及对应位点配对。

---

### 8. 理论结果（逻辑）
在 GW 代价满足局部强凸近似时，**拍卖迭代为收缩映射**，在有限步内收敛到近优 \( \varepsilon \)-匹配。给出**冲突率上界与分配误差**之间的关系式：  
$$
\mathrm{OverlapRate} \;\le\; f\!\left(\varepsilon,\; \mathrm{SDF}_{\text{margin}}\right),
$$
其中 \( f \) 单调递增。

---

### 9. 复现实验设计（MFE）
- **Full**：2×A100-80GB，构建多配体 PDB 组合，预算 ≤ 150 GPUh。  
- **Lite**：2–3×4080，\(N \le 3\) 情形，预算 ≤ 36–60 GPUh。  
- **实现细节**：GW 近似用小批次近似 + 行并行拍卖实现（CPU/GPU 混合）。

---

### 10. 负结果利用
若分配层无效，输出分配热力映射 \( \Gamma \) 并把某些口袋分类为 “OT-不适” 类，形成多配体对接**可用/不可用**的分层评测。

---

### 11. 跨学科价值
引入 OT / 拍卖机制到分子对接，为**多体占位/片段组装**问题提供可证伪机制层，能推广到**配体库拼接与药代学复合系统**。

---

### 12. 投稿与博士论文映射
- **算法/机制**：ICLR / NeurIPS  
- **应用/评测**：Bioinformatics / Information Sciences (IS)  
- **博士论文**：方法篇（多体机制）、应用篇（多配体评测）

---

### 13. 递进关系
与 Idea-A（IMF）配合：先做单体合法化，再做多体全局分配。

