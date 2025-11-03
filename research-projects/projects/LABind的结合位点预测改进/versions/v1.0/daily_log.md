# 🗓️ {2025-11-04} — Version {V1.0} 工作日志

## 🧭 今日主题
> 本日聚焦的核心任务或实验方向  
> 例如：Fisher-Z 正则稳定性实验、Copula-NLL 梯度分析、Curvature 模块验证等。

---

## ✅ 今日进展
- [x] （代码）实现 Copula-NLL loss 子模块；
- [x] （实验）完成 PDBBind 1000 条子集训练；
- [ ] （分析）验证 ρ-gradient 平滑效果；

---

## 💡 思考与改进想法
- ρ 的梯度震荡可能来自 loss 缩放不均衡；
- 考虑加入 `torch.tanh` 平滑替代 Fisher-Z；
- 也许可以通过 batch-level normalization 稳定 σ_r；
- **思路关键点**：引入 curvature-aware alignment 有望进一步统一潜空间。

---

## 🧪 实验记录
| 实验编号 | 数据集 | 参数修改 | 结果摘要 |
|-----------|----------|------------|-------------|
| exp_01 | CrossDocked | λ_align=0.2 | RMSD <2Å: 70.1%, Spearman ρ: 0.84 |
| exp_02 | PDBBind | ρ_init=0.3 | 梯度震荡减弱，稳定性↑ |

📊 可视化：
![Loss Curve](../assets/figures/{DATE}_loss_curve.png)

---

## 📈 临时观察
> 训练到 epoch 80 时 loss 出现震荡；  
> ρ 在 [0.55, 0.65] 区间波动，Copula-NLL 的稳定性受 batch_size=4 影响明显。

---

## 🪶 明日计划
- [ ] 验证 Fisher-Z vs tanh 在 ρ 学习上的稳定性；
- [ ] 复现 curvature 正则实验；
- [ ] 更新 version log.md 中的改进摘要。

---

## 🧠 备注 / 灵感
> “多任务协同的核心不是共享 backbone，而是共享不确定性结构。”
