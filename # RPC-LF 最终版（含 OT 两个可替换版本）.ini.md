# RPC-LF 最终版（含 OT 两个可替换版本）
> **Robust Pocket Conditioning with Learnable & Fair Radius**  
> 软路由 + 覆盖匹配半径（隐式可导）+ 部署上界收紧（鲁棒/OT）  
> 目标：**仅用预测口袋推理**时，构象精度尽量逼近“GT 全蛋白推理”上限；训练/推理算子一致、低开销、可解释、可复现。

---

## 0. 研究问题与细化痛点

**总体问题**：两阶段“口袋预测→构象预测”中，推理期只能用**预测口袋**；而常规做法在训练/评测中混用**GT 口袋或硬裁剪**，导致部署时性能与稳定性下降。

**痛点（3 点）**  
1. **硬裁剪放大误差**：固定半径/比例的硬口袋裁剪使边界不光滑；中心误差的微小变化可引起支持集突变（<2Å 指标敏感）。  
2. **训推错配**：训练常混入 GT 口袋或启发式 schedule，推理只靠预测口袋；目标与部署不一致。  
3. **半径不可解释/难校准**：直接回归或经验半径依赖后处理（采样、置信重排），复杂且不稳。

---

## 1. 变量与符号（逐一精确定义）

- 蛋白残基坐标：$\{r_j\in\mathbb{R}^3\}_{j=1}^N$  
- 配体 GT 姿势原子集合：$Y$  
- 口袋中心：训练/推理用预测中心 $\hat c$（上界对照可用 $c^{*}$）  
- 主干网络：构象预测 $f_\theta(\cdot)$（可为 FABind / FABind+ / PPDock pose/scorer 头）  
- 任务损失：$\ell\!\left(f_\theta(\cdot),Y\right)$（可为平滑 RMSD/接触/距离图等）  
- Sigmoid：$\sigma(x)=1/(1+e^{-x})$  
- 平滑带宽：$\tau>0$（推荐 $1.0\,\mathrm{\AA}$）  
- 软掩码（软路由）：  
  $$\alpha_{c,R,\tau}(r_j)=\sigma\!\left(\frac{R-\lVert r_j-c\rVert_2}{\tau}\right)\in(0,1)$$
- GT 覆盖软标签（由 $Y$ 自动构造）：  
  $$d_j^{\mathrm{GT}}=\min_{a\in Y}\lVert r_j-a\rVert_2,\quad
    q_j=\sigma\!\left(\frac{r_{\mathrm{cut}}-d_j^{\mathrm{GT}}}{\gamma}\right),\quad
    B=\sum_{j=1}^N q_j,\ \ p=\tfrac{B}{N}$$
  推荐常量：$r_{\mathrm{cut}}=5.0\,\mathrm{\AA}$，$\gamma=1.0\,\mathrm{\AA}$  
- 软路由总质量：$\lVert\alpha\rVert_1=\sum_{j=1}^N \alpha_j$  
- 归一软权：$\tilde\alpha=\alpha\,\dfrac{B}{\lVert\alpha\rVert_1}$  
- 中心误差常量（训练集分位数）：$\delta$（如 75% 分位）  
- 覆盖比例预测器（推理期用）：$s_\theta(P,L,c)\in(0,1)$，经标量校准得 $\widehat p$

---

## 2. 模块与核心公式

### 2.1 软路由注入（替代硬裁剪；不改主干拓扑）
对所有配体→蛋白 cross-attention 层（Key 侧）将 $\log\alpha$ 加到 logits：
$$
S^{(\ell)}_{ij}=\frac{Q^{(\ell)}_i\!\cdot\!K^{(\ell)}_j}{\sqrt d}+b^{(\ell)}_{ij},\quad
\boxed{\ \widetilde S^{(\ell)}_{ij}=S^{(\ell)}_{ij}+\log\big(\mathrm{clamp}(\alpha_j,\varepsilon,1)\big)\ }
$$
随后 softmax 归一得到注意力；self-attention 不改。

### 2.2 覆盖匹配半径 $R*$（唯一解；前向二分；隐式可导）
令单调方程
$$
h(R,\theta)=\sum_{j=1}^N \alpha_{c,R,\tau}(r_j;\theta)-B=0
$$
其根 $R^*(\theta)$ **唯一存在**；前向用二分/分位数近似求解；反传用隐式微分
$$
\frac{\partial R^*}{\partial \theta}
= - \frac{\partial h/\partial \theta}{\partial h/\partial R},\qquad
\frac{\partial h}{\partial R}=\frac{1}{\tau}\sum_{j=1}^N \alpha_j(1-\alpha_j)\ >0
$$
实现上可对 $\partial h/\partial R$ 执行 `detach` 以避免二阶图膨胀。

### 2.3 覆盖对齐项（Base: KL；OT-1: $\mathsf{W}_1$）
- **Base（KL）**：
$$
\mathcal L_{\mathrm{cov}}^{\mathrm{KL}}
= \mathrm{KL}\!\left(\frac{\tilde\alpha+\varepsilon}{B+\varepsilon N}\ \middle\|\ \frac{q+\varepsilon}{B+\varepsilon N}\right)
$$
- **OT-1（一维 Wasserstein）**：以径向距离 $d_j=\lVert r_j-c\rVert$ 为地度量，将 $P=\tilde\alpha/B$、$Q=q/B$ 定义在离散点集 $\{d_j\}$ 上：
$$
\mathcal L_{\mathrm{cov}}^{\mathsf{W}_1}
= \mathsf{W}_1(P,Q)=\int \big|F_P(r)-F_Q(r)\big|\,dr
$$
> 注：一维 OT 的最优必要充分条件是 CDF 对齐；其一阶条件正是 $\sum_j\alpha(R)=B$，与 $R^*$ 方程一致（几何一致性）。

### 2.4 部署上界收紧（Base: 一阶；OT-2: Wasserstein-DRO）
- **Base（一阶上界的可微化）**：对任意可微 $\ell$，
$$
\sup_{\lVert\Delta c\rVert\le\delta}\ell(c+\Delta c)\ \le\ \ell(c)+\delta\,\lVert\nabla_c\ell(c)\rVert_2
$$
选取平方稳定形（单次前向，二阶在反传期隐式出现）：
$$
\boxed{\ \mathcal L_{\mathrm{rob}}^{\mathrm{grad}}
  = \frac{\lambda_{\mathrm{rob}}}{2}\,\delta^2\,
  \big\lVert\nabla_c\,\ell\big(f_\theta(\alpha_{c,R^*,\tau}),Y\big)\big\rVert_2^2\ }
$$
- **OT-2（Wasserstein-DRO 对偶）**：在以中心位移为地度量的 Wasserstein 球 $\mathbb B_\rho$ 上做最坏期望
$$
\min_\theta\ \sup_{\mathbb Q:\ W(\mathbb Q,\widehat{\mathbb P})\le \rho}\ 
\mathbb E_{\mathbb Q}\!\left[\ell\big(f_\theta(\alpha_{c,R^*,\tau}),Y\big)\right]
$$
对 Lipschitz-$\ell$（或局部线性化）有强对偶：
$$
\mathcal L_{\mathrm{rob}}^{\mathrm{DRO}}
\ \approx\ \mathbb E_{\widehat{\mathbb P}}[\ell]\ +\ \rho\cdot \mathrm{Lip}_c(\ell)
\ \ \leadsto\ \ \text{以}\ \lVert\nabla_c\ell\rVert\ \text{为罚项的实现}
$$
> 实操上可直接用 $\mathcal L_{\mathrm{rob}}^{\mathrm{grad}}$，并以 DRO 对偶作为理论背书；若需数值 OT，可在小网格上用熵正则 Sinkhorn 近似 $\sup$（通常不必）。

---

## 3. 总损失（Base 与两个 OT 版本）

- **RPC-LF（Base）**  
$$
\boxed{\ 
\mathcal L_{\mathrm{RPC\!-\!LF}}^{\mathrm{Base}}
= \underbrace{\ell\big(f_\theta(\alpha_{c,R^*,\tau}),Y\big)}_{\text{任务}}
\ +\ \lambda_{\mathrm{cov}}\ \underbrace{\mathcal L_{\mathrm{cov}}^{\mathrm{KL}}}_{\text{覆盖对齐}}
\ +\ \underbrace{\mathcal L_{\mathrm{rob}}^{\mathrm{grad}}}_{\text{部署上界收紧}}\ }
$$

- **RPC-LF（OT-1：Coverage-OT）**  
$$
\boxed{\ 
\mathcal L_{\mathrm{RPC\!-\!LF}}^{\mathsf{OT\!-\!1}}
= \ell\big(f_\theta(\alpha_{c,R^*,\tau}),Y\big)
\ +\ \lambda_{\mathrm{cov}}\ \mathcal L_{\mathrm{cov}}^{\mathsf{W}_1}
\ +\ \mathcal L_{\mathrm{rob}}^{\mathrm{grad}}\ }
$$

- **RPC-LF（OT-2：DRO-OT）**  
$$
\boxed{\ 
\mathcal L_{\mathrm{RPC\!-\!LF}}^{\mathsf{OT\!-\!2}}
= \ell\big(f_\theta(\alpha_{c,R^*,\tau}),Y\big)
\ +\ \lambda_{\mathrm{cov}}\ \big(\mathcal L_{\mathrm{cov}}^{\mathrm{KL}}\ \text{或}\ \mathcal L_{\mathrm{cov}}^{\mathsf{W}_1}\big)
\ +\ \mathcal L_{\mathrm{rob}}^{\mathrm{DRO}}\ }
$$

**推荐常量**：$\tau=1.0\,\mathrm{\AA}$、$\lambda_{\mathrm{cov}}=0.5$、$\lambda_{\mathrm{rob}}=0.1\sim0.2$、$\delta=$ 训练集中心误差 75% 分位；$\varepsilon\in[10^{-6},10^{-4}]$。

---

## 4. 训练与推理步骤（按部就班即可落地）

### 4.1 训练（单前向；不跑 GT 分支）
1. **弱标签构造**：由 $Y$ 计算 $q_j$、$B=\sum q_j$、$p=B/N$。  
2. **中心与半径**：中心用**预测** $\hat c$；计算 $d_j=\lVert r_j-\hat c\rVert$，二分求 $R^*$ 使 $\sum\alpha=B$；固定 $\tau=1.0\,\mathrm{\AA}$（或下限）。  
3. **软路由注入**：在所有 cross-attn 层 logits 加 $\log\alpha$。  
4. **前向与主损**：得到 $\ell\!\left(f_\theta(\alpha_{c,R^*,\tau}),Y\right)$。  
5. **覆盖对齐**：取 $\mathcal L_{\mathrm{cov}}^{\mathrm{KL}}$（或 $\mathsf{W}_1$）。  
6. **鲁棒项**：令 `center.requires_grad_(True)`，取 $g=\nabla_c\ell$（`create_graph=True`），记 $\mathcal L_{\mathrm{rob}}^{\mathrm{grad}}=\tfrac12\lambda_{\mathrm{rob}}\delta^2\lVert g\rVert_2^2$。  
7. **合成与更新**：按所选版本组合总损失，反传更新。  
8. **日志与验证（建议）**：记录 $\lVert g\rVert$ 分布、按中心误差分桶的 “GT 全图 vs Pred 软路由” gap 收紧曲线、ECE/NLL 下降。

### 4.2 推理（同算子；零额外前向）
1. 口袋中心 $\hat c$ 与覆盖预测 $s_\theta$（一层 MLP 即可）。  
2. 验证集做一次**标量温度缩放/等分位校准**，得 $\widehat p\in[p_{\min},p_{\max}]$（如 $[0.05,0.25]$）。  
3. 解 $\sum\alpha=\widehat p N$ 得 $R^*$，在 logits 加 $\log\alpha$，输出构象/评分。

> **Oracle 上界（对照，不可部署）**：可用 $c^*$ 或 $B$ 替换 $\hat c$、$\widehat p$，评估理论最优性能上限。

---

## 5. 代码（PyTorch-like，可直接改 FABind）

> 说明：在 FABind 的配体→蛋白 cross-attn 中对每层 logits 做 `+ alpha_log[None,:]`；self-attn 不动。以下示例含**半径二分**、**KL 覆盖**、**一阶鲁棒项**、**1D-$\,\mathsf{W}_1$**。

```python
import torch
import torch.nn.functional as F

def soft_mask(dist, R, tau):
    # dist: (N,), R,tau: scalar tensors/floats
    return torch.sigmoid((R - dist) / tau).clamp_min(1e-4)

@torch.no_grad()
def solve_radius_bisect(dist, B, tau, iters=20):
    # dist: (N,), B: scalar mass, tau: float
    lo = torch.tensor(0.0, device=dist.device)
    hi = torch.tensor(dist.max().item() + 20.0, device=dist.device)
    for _ in range(iters):
        mid = (lo + hi) * 0.5
        mass = soft_mask(dist, mid, tau).sum()
        lo, hi = torch.where(mass < B, lo, mid), torch.where(mass < B, mid, hi)
    return ((lo + hi) * 0.5).detach()

def kl_cover(alpha, q, eps=1e-6):
    P = (alpha / alpha.sum()).clamp_min(eps)
    Q = (q     / q.sum()).clamp_min(eps)
    return (P * (P / Q).log()).sum()

def w1_1d_radial(dist, wP, wQ, eps=1e-12):
    # dist: (N,), w*: (N,), sum to 1
    vals, idx = torch.sort(dist)
    P = wP[idx].clamp_min(eps)
    Q = wQ[idx].clamp_min(eps)
    cP = torch.cumsum(P, dim=0)
    cQ = torch.cumsum(Q, dim=0)
    dr = torch.diff(vals, prepend=vals[:1])
    return (torch.abs(cP - cQ) * dr).sum()

def rpc_lf_loss_base(
    protein_xyz, center_pred, q_soft,
    pocket_net, pose_net,          # your existing modules
    tau=1.0, lambda_cov=0.5, lambda_rob=0.1, delta_const=4.0
):
    # 1) target mass
    B = q_soft.sum()

    # 2) distances & radius
    dist = (protein_xyz - center_pred[None,:]).norm(dim=-1)  # (N,)
    R_star = solve_radius_bisect(dist, B=B, tau=tau)
    alpha  = soft_mask(dist, R_star, tau)
    alpha_log = F.layer_norm(alpha.log(), normalized_shape=alpha.shape)

    # 3) inject log(alpha) to ALL ligand->protein cross-attn layers (implement inside pocket_net)
    pocket_net.set_protein_bias(alpha_log)  # two-line change inside attention

    # 4) forward pose & task loss
    pose_out = pose_net(pocket_net)         # follow your codebase API
    loss_pose = pose_out["loss"]

    # 5) coverage alignment (KL)
    loss_cov = kl_cover(alpha, q_soft)

    # 6) robustness: single forward, 2nd-order only in backward graph
    center_pred.requires_grad_(True)
    g = torch.autograd.grad(loss_pose, center_pred, create_graph=True)[0]
    loss_rob = 0.5 * lambda_rob * (delta_const ** 2) * (g.pow(2).sum())

    # 7) total
    loss = loss_pose + lambda_cov * loss_cov + loss_rob
    with torch.no_grad():
        logs = dict(loss_pose=loss_pose, loss_cov=loss_cov, loss_rob=loss_rob, R_star=R_star)
    return loss, logs

def rpc_lf_loss_ot1(
    protein_xyz, center_pred, q_soft,
    pocket_net, pose_net,
    tau=1.0, lambda_cov=0.5, lambda_rob=0.1, delta_const=4.0
):
    B = q_soft.sum()
    dist = (protein_xyz - center_pred[None,:]).norm(dim=-1)
    R_star = solve_radius_bisect(dist, B=B, tau=tau)
    alpha  = soft_mask(dist, R_star, tau)
    alpha_log = F.layer_norm(alpha.log(), normalized_shape=alpha.shape)
    pocket_net.set_protein_bias(alpha_log)

    pose_out = pose_net(pocket_net)
    loss_pose = pose_out["loss"]

    P = (alpha / alpha.sum()).clamp_min(1e-8)
    Q = (q_soft / q_soft.sum()).clamp_min(1e-8)
    loss_cov = w1_1d_radial(dist, P, Q)

    center_pred.requires_grad_(True)
    g = torch.autograd.grad(loss_pose, center_pred, create_graph=True)[0]
    loss_rob = 0.5 * lambda_rob * (delta_const ** 2) * (g.pow(2).sum())

    loss = loss_pose + lambda_cov * loss_cov + loss_rob
    with torch.no_grad():
        logs = dict(loss_pose=loss_pose, loss_cov=loss_cov, loss_rob=loss_rob, R_star=R_star)
    return loss, logs

```

## 6. 对比改进点
- （相对 FABind / FABind+ / PPDock）
- vs FABind（20Å 硬裁剪 + schedule）：
硬裁剪→软路由（可导、抗边界抖动）；固定 20Å→覆盖匹配半径 $R^*$（唯一解、可导、可校准）；
schedule→部署上界代理（$\lVert\nabla_c\ell\rVert$ 或 DRO-对偶）。
预期：在主体误差桶（$\delta\le 4,\mathrm{\AA}$），“GT 全图 vs Pred 软路由” gap 显著收紧；Top-1 <2Å 稳定 +3%（绝对）；ECE/NLL 下降。

- vs FABind+（动态半径/覆盖分析/多采样）：
经验回归半径→覆盖匹配的一维方程（几何一致 + 唯一解 + 隐式可导）；
去多采样/置信重排→单前向 + 二分（更简洁、算力更低）；
若用 OT-1：覆盖对齐从 KL→$\mathsf{W}_1$（惩罚“搬多少 + 搬多远”）。
预期：同预算/统一口径下持平或小幅领先（尤其在 $2!\sim!4,\mathrm{\AA}$ 桶），且更易复现。

- vs PPDock（anchor + 裁 15%）：
“15%”视作 $p=0.15$ 特例→覆盖比例可学/可校准；
硬比例裁切→软路由 + 上界收紧。
预期：浅口袋/边界样本显著受益，Top-1 <2Å ≥ +4%（绝对）。

效率：训练时间 ≤ +8–10%（排序/二分 + 一阶梯度），显存≈持平；推理近 0% 增量

## 7. 评测与复现（公平协议）

公平对比：只比 GT 全蛋白（不裁剪） vs Pred 软路由（本方法）；杜绝“GT+半径”的人造基线。

OOD：family / geometry / scaffold 及交集；指标：Top-1/5 <2Å/<5Å、RMSD、ECE、NLL。

误差分桶曲线：按中心误差 $\delta$ 分桶（<2Å、2–4Å、4–8Å、≥8Å）报告 “GT vs Pred” gap 收紧。

覆盖–RMSD 曲线：整体右移（覆盖↑、RMSD↓）。

效率剖面：训练/推理耗时与显存 profile 佐证“单前向、低开销”。

## 8. 超参数（可直接采用）

$\tau=1.0,\mathrm{\AA}$，$\lambda_{\mathrm{cov}}=0.5$，$\lambda_{\mathrm{rob}}=0.1\sim0.2$，

$\delta=$ 训练集中心误差 75% 分位（常量），

覆盖校准区间 $[p_{\min},p_{\max}]=[0.05,0.25]$，

数值保护 $\varepsilon=10^{-6}\sim10^{-4}$。

## 总结
RPC-LF：用软路由 + 覆盖匹配半径（唯一解、隐式可导） + 部署上界收紧统一替换口袋裁剪与训推错配；OT-1 让覆盖对齐与几何一致，OT-2 为鲁棒项提供 Wasserstein-DRO 对偶背书。方案单前向、低开销、可解释、易复现，可直接插入 FABind / FABind+ / PPDock。