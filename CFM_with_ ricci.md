# 问题背景
扩散/流匹配的分子对接模型在路径径生成中，高曲率狭窄口袋区域易震荡、误配与过度采样。

# 范式局限
- **现有范式**：现有扩散/流匹配在扩散过程种把几何当静态条件，忽略口袋表面的曲率演化，会在高曲率（凹槽/尖凸/鞍点）附近引发各向同性采样的系统性失配：轨迹震荡、碰撞修复激增、探索效率骤降、排序与校准失真，尤其在 OOD + 低采样预算 时被放大。
- **具体后果**：未把口袋曲率的演化/单调结构并入生成动力学，在高曲率、狭窄沟槽处，噪声与步长各向同性 → 轨迹震荡、碰撞率上升。 
1. **窄凹槽（高负曲率）中的轨迹震荡/贴壁–反弹循环**  
   - **后果**：各向同性步长/噪声不随几何调节，轨迹在狭窄沟槽里反复撞壁，来自抖动，探索效率骤降。  
   - **征兆**：单位轨迹软碰撞计数持续上升；$\Pr[\Delta E_K > 0]$ 上升；同等步数下**有效位移/步**下降。  
   - **指标**：Top-1 < 2Å、Top-5 < 2Å 下降；**效率前沿**（采样/秒数/显存→命中率）AUC 后退。  

2. **凸起区域的硬冲突与不可行样本堆积**  
   - **后果**：缺少沿 $\nabla K$ 的回避，引发到探索/突攀，大量假选点靠后处理剔除。  
   - **特性**：**最小原子间距分布拉长（长尾）**；碰撞体复刻比例占比↑；PoseBusters 类检查不通过率↑。  
   - **指标**：同预算下推理时长与显存值上升；最终 DCC 与 **<5Å 命中率** 受拖累。  

3. **OOD 几何分布移位导致校准失真与排序失衡**  
   - **后果**：训练期静态几何编码与测试口袋曲率统计不匹配，生成阶段不自适应 → 置信度与真实误差脱钩。  
   - **征兆**：ECE/NLL 明显恶化；**风险–覆盖曲线**显著后陡降；打分–RMSD 相关性在高曲率桶显著变差。  
   - **指标**：ECE↑ / NLL↑；虚筛 EF@1%/BEDROC 下降；Top-1 提升需要“堆采样”才能弥补。

# 具体主张
- **概述**：
    1. **机制**：我们将几何写入生成动力学，使用切向曲率引导作为能量  $V_K = \mathbb{E}[K]$ 的最陡下降步，并以正均曲率单调与 SDF 屏障提供可观测的稳定与安全约束。  三者共同在不改变模型结构的前提下降低震荡与碰撞。


    1. **理论**：证明两定理一推论（能量最陡下降且等变；凸性单调+Fisher 可识别；α→0 残余效应上界），并量化 E(3) 的离散误差。

    2. **证据**：提供**可识别性响应曲线**、**$\alpha$ 衰减收敛**与**最坏复杂度**三类新实验，附**自动触发器与日志**，保证可复现。



1. **现有范式的“静态几何”假设**：  
   几何只在条件编码或得分器里使用；生成 ODE/SDE 本体仍是**几何无关**的各向同性抗动力与学习到的速度/噪声。结果是 **局部地形（凹/凸/鞍）** 不影响采样步态。

2. **在 OOD + 低采样下的失效征兆（可度量）**：  
   - **震荡/漂移**：沿轨迹的  
     $$
     \Delta E_{K}(k) = \mathbb{E}[K(x_k)] - \mathbb{E}[K(x_{k-1})]
     $$  
     频繁为正； 表面/体场与可导映射，其中
     取蛋白 SES 三角网格 $\mathcal{M} = (V, F)$，每顶点预存 $(K_v, H_v, \kappa_{1,v}, \kappa_{2,v}, n_v)$；三角面内用**重心插值**得连续场 $\tilde{K}(p), \tilde{H}(p), \tilde{n}(p)$。   预计算有界平滑 SDF $\phi : \mathbb{R}^3 \to \mathbb{R}$（外部为正），其梯度 $\nabla \phi(r) = n(p)$ 在一阶可导。  对给定位姿 $x$ 的配体原子 $\{r_i(x)\}_{i=1}^N$，用最近点投影 $p_i = \Pi_{\mathcal{M}}(r_i)$，并以 SDF 构造**可导接触权重**：
      $$
      \rho_i(x) = 
      \frac{\exp\!\big(-\tfrac{\phi(r_i)^2}{\tau^2}\big)\cdot \mathbf{1}\!\big(\phi(r_i) \le d_{\max}\big)}
           {\sum_j \exp\!\big(-\tfrac{\phi(r_j)^2}{\tau^2}\big)\cdot \mathbf{1}\!\big(\phi(r_j) \le d_{\max}\big)} .
      $$
      $\rho_i(x)$ 采用平滑窗与归一化，对 $x$ 一阶可导；  阈值 $\mathbf{1}(\phi \le d_{\max})$ 用平滑近似实现以保持可微。
       - **期望算子定义**（统计域→软接触原子）：
      $$
      \mathbb{E}_x[K] = \sum_i \rho_i(x)\,\tilde{K}(p_i), 
      \qquad
      \mathbb{E}_x[H] = \sum_i \rho_i(x)\,\tilde{H}(p_i).
      $$
    
      以上对 $x$ 可微：$\partial \mathbb{E}_x[\cdot]/\partial x$ 通过 $\phi, \Pi_\mathcal{M}$ 与重心坐标模式求导得到。 落 地参数 $\tau = 1.0\ \text{Å},\quad d_{\max} = 3.0\ \text{Å}$  SDF 用窄带（5Å）+ 三线性插值；最近点查询用 BVH/Embree  所有张量在 GPU 上缓存。

   - **碰撞率上升**：单位轨迹逆软碰撞项  
     $\sum_{i,j} \exp(-(d_{ij} - d_0)/\sigma)$  
     的累计值上升；  
   - **命中/校准双降**：Top-1<2Å 与 ECE/NLL 同时恶化；  
   - **效率前沿后退**：在同等采样数/显存/秒数下，前沿 AUC 更低。  

3. **导致失效的直接机制：几何各向同性误配** —— 在高负曲率凹槽处仍按均匀噪声扩散，步长过大引发“贴壁—反弹—再贴壁”的循环；在正曲率突起处回避不足，造成反复碰撞修正。  

4. ## 训练目标（仅 3 项）

    $$
    \mathcal{L} \;=\; \mathcal{L}_{\mathrm{CFM}}
    + \lambda \,\mathrm{ReLU}\!\big(\mathbb{E}[H_+]_k - \mathbb{E}[H_+]_{k-1}\big)
    + \beta \,\mathbb{E}\!\left[\mathrm{softplus}(\varepsilon - \phi)^2\right]
    $$  
    
    - 第一项：照基线  
    - 第二项：$L_c^{(+)}=\lambda·ReLU(ΔE[H_+])$ 
    - 第三项：软屏障  
    
    
    
    ## 引导项（1 行）
    
    $$
    x_{k+1} = x_k + \Delta t \Big[f_\theta(x_k,t) 
    + \alpha(t)\,\Pi(\nabla_S K)\Big] + \mathrm{noise}(t)
    $$  
    
    
    ### 说明
    - **Π**：把每原子方向场按雅可比最小二乘投到 {平移, 旋转, 扭转}，可直接用 AD 求雅可比。    



   其中 $\Pi_{x}$ 代表面梯度场映射成对配体**平移/旋转/扭转**的引导；  
   $L_c$ 强化“期望曲率非增”的**耗散结构**，稳定轨迹。
   我们把表面方向场变成**原子虚拟力**，再用雅可比把虚拟力投影到位姿 DOF（平移/旋转/各扭转角）。
    
    
    ### 1. 原子级几何引导只保留切向引导以降复杂度
    
    对每个接触原子 $i$：
    

    
    - **切向**：
      $$
      f_i^{(t)} =\,P_T(p_i)\,\nabla \tilde{K}(p_i),
      $$
      其中 $P_T = I - \tilde{n}\tilde{n}^\top$，$\nabla_S$ 为表面梯度。  
    

    
    ---
    
    ### 2. DOF 投影（最小二乘）
    
    记配体原子位置由 DOF $x=[t, \omega, \{\theta_b\}]$ 给出，运动学映射为 $r(x)$。  
    令雅可比矩阵 $J = \partial r/\partial x$（可由自动微分得到）。  
    把虚拟力 $F = [f_1; \ldots; f_N]$ 投影到 DOF：
    
    $$
    \Delta x = (J^\top W J)^{-1} J^\top W F, 
    \qquad 
    v_{\text{geom}} = \alpha(t)\,\frac{\Delta x}{\Delta t}.
    $$
    
    权重 $W = \mathrm{diag}(w_i I_3)$，$w_i \propto \rho_i$（接触越强权重越大）。
    
    ---
    
    ### 等变性
    
    若对 $(r,F)$ 同时施加全局刚体变换 $g \in E(3)$，则 $J \mapsto RJ,\ F \mapsto RF$，从而 $\Delta x$ 不变，保证等变性。
    
    实现提示：用最小二乘将原子方向场投影到 {平移、旋转、扭转}，该步骤 E(3) 等变、复杂度线性；
          
    ## 单一$H_+$耗散条款
    
    ### 1. 只对“凸性风险”做耗散
    把耗散对象从 $K$ 改为**正均曲率**：  
    $$
    H_+(p) = \max(\tilde{H}(p), 0).
    $$  
    
    耗散正则定义为：  
    $$
    L_c^{(+)} = \lambda \cdot \frac{1}{T} \sum_{k=1}^T 
    \mathrm{ReLU}\!\Big( \mathbb{E}_{x_k}[H_+] - \mathbb{E}_{x_{k-1}}[H_+] \Big).
    $$  
    
    👉 只压制“越向外凸越多”的趋势，不限制进入凹槽（$\tilde{H} < 0$）。
    
   
   
    ### 3. 与 SDF 屏障配合，明确“回避 vs 进入”的方向性
    
    加入软屏障：  
    $$
    B(\phi) = \mathrm{softplus}(\epsilon-\phi)^2,
    $$  
    总损失：  
    $$
    \mathcal{L} = \mathcal{L}_{\mathrm{CFM}} + \lambda L_c^{(+)} + \beta\, \mathbb{E}_x[B(\phi)].
    $$  
    
    👉 避免穿模：进入凹域靠切向梯度，回避凸起靠 $H_+$ 耗散 + SDF 屏障，分工清晰。


# 创新出发
A,算法交互 B 理论约束

## A 算法交互（做什么、在哪里、如何落地）

- **做什么**：  
  把口袋的曲率梯度 $\nabla K$ 作为几何势，直接写入**目标速度场**并参与反传； 减少早期引导噪声。  

- **在哪里**：  
  发生在**生成器本体**（扩散/流匹配的 ODE/SDE 或速度场），而非事后重排序；训练与推理**同构**（推理也用 $\alpha(t)$ 引导）。  

- **如何落地**：  
  提供可复用接口 `query_curv(x)->{K, \nabla K}` 与 `project_to_dofs(\nabla K, x)`；  
  默认超参：$\alpha_0 = 0.35$ 线性退火至 0；$\beta = 0.1$（软碰撞）。  

- **与“后验打分”的区别**：  
  后者只调排序，不改变样本**生成分布**；本方案改变量场几何，在同等预算下降低对“多样化暴力采样”的依赖。

## B 理论约束（做什么、可观测、何以保障稳定）

- **做什么**：  
  在损失中加入**曲率耗散正则** $L_c$，要求沿轨迹**期望曲率非增**（分段）；这把 Ricci/高斯曲率的几何单调性转化成**可反传的稳定约束**。

- **可观测**：  
  训练时实时记录 $\Delta E_{K}(k)$ 的负比例、单位轨迹**软碰撞计数**、风险–覆盖曲线 **AUC**；并将这些与 **Top-1 < 2\AA**、**ECE/NLL** 联动可视化，作为**失效早期预警**。  
  其中 $\displaystyle \Delta E_{K}(k)=\mathbb{E}[K(x_k)]-\mathbb{E}[K(x_{k-1})]$。

- **稳定性直觉**：  
  若 $\mathbb{E}[K]$ 非增，则“贴壁—反弹”模式被抑制；在步长满足条件时，单位长度的**碰撞率/漂移**被上界为 $\mathcal{O}(\alpha_{\max})+\mathcal{O}(\eta)$（详见最小理论结果）。


# 系统流程
1. 口袋网格与曲率估计  
2. 曲率感知条件流匹配  


# 模块设计
### M1 口袋网格与曲率估计
- **作用**：从口袋表面三角网格估计离散高斯/近似 Ricci 曲率 K 与 ∇K
- **输入**：蛋白坐标/半径，口袋掩码/网格  
- **输出**：{K, ∇K}网格索引  
- **必要性**：提供几何先验与引导势；为 CFM 与细化共享  

### M2 曲率感知 CFM（K-FlowMatch）
- **作用**：目标速度场 vθ(x,t|protein) ← vθ + α∇K；加入曲率耗散正则 L_c 约束 E[K(x(t))] 单调  
- **输入**：配体原子坐标/扭转，蛋白图特征，{K, ∇K}, t  
- **输出**：速度场/流轨迹，候选姿态集  
- **必要性**：在生成阶段抑制震荡、聚焦凹陷通道，降低采样需求  


# 理论结果
- 理论顺序：变分最陡下降（定理A）→ 等变/单调/可识别（定理B + 误差界）→ α→0 收敛（推论C）→ 复杂度与落地。 
## 记号与假设

- **口袋网格**：$\mathcal{M} = (V,F)$，重心插值得 $\tilde{K}, \tilde{H}, \tilde{n}$；  
  SDF $\phi$ 在窄带内 $C^1$。  

- **接触权重**：$\rho_i(x)$ 平滑、归一化；期望 $\mathbb{E}_x[\cdot] = \sum_i \rho_i(x)(\cdot)$ 对 $x$ 一阶可导。  

- **生成向量场**：$f_\theta$ **one-sided-Lipschitz**，步长 $\eta \le 1/L_\theta$。  

- **投影度量**：$G(x) = J^\top W J$ 正定；  
  $W = \mathrm{diag}(w_i I_3),\ w_i \propto \rho_i$。

## 定理 A（多步最陡下降 + E(3) 等变）

令 $V_K(x) = \mathbb{E}_x[K],\quad F_t=\{P_T\nabla_S \tilde{K}(p_i)\}_i$，则  

$$
\nabla_x V_K(x) = J^\top W F_t, 
\qquad 
\Delta x^\ast = \arg\min_{\Delta x}\Big[-\langle \nabla_x V_K,\Delta x\rangle + \tfrac{1}{2}\|\Delta x\|_G^2 \Big] 
= G^{-1}\nabla_x V_K.
$$  

从而  
$$
\Pi(\nabla_S K) = (J^\top W J)^{-1} J^\top W F_t .
$$  

对任意 $g \in E(3)$ 共同作用下，$(J,F)\mapsto RJ, RF \;\Rightarrow\; \Pi(\nabla_S K)$ 不变。  

---

### 命题 A-1（离散误差上界）

若 $h,\delta$ 网格步长，SDF 插值误差 $\delta$，则  

$$
\|\Delta x(g\cdot x) - \Delta x(x)\| \le C_1 h + C_2 \delta^2.
$$  

---

### 命题 A-2（映射扰动非奇）

若 $\|e_x\|_\infty \le \epsilon_x,\ \|e_w\|_\infty \le \epsilon_w$，则  

$$
\|\Delta x^\ast - \hat{\Delta x}^\ast\| \le \|G^{-1}\| \big(\|J^\top W\| e_g + \eta(J,We_x)\big).
$$  

---

## 定理 B（凸性单调 + Fisher 可识别）

用 $H_+(p)=\max(\tilde{H}(p),0)$ 定义，建立 Lyapunov  

$$
V(x) = \mathbb{E}_x[H_+] + \mu \mathbb{E}_x[B(\phi)], 
\qquad 
B(\phi)=\mathrm{softplus}(-\phi^2).
$$  

若 SDF 等带 Lipschitz 且 $\eta \le 1/L$，存在常数 $c>0$ 使  

$$
\Delta \mathbb{E}[H_+] \le 0, 
\qquad 
\mathbb{E}[B(\phi_{k+1})] - \mathbb{E}[B(\phi_k)] \le -c\beta\eta + C_1 \alpha_{\max}\eta^2.
$$  

设可观测统计 $S(\theta)=[S_1,S_2,S_3]$，其中 $S_1=\partial_t \mathbb{E}[H_+],\ S_2=\mathbb{E}[B(\phi)],\ S_3=\|\Pi(\nabla_S K)\|_2$，观测噪声  
$S_k^{\text{obs}}=S_k+\xi_k,\ \xi_k\sim\mathcal{N}(0,\Sigma)$，则经验 Fisher  

$$
F=\sum_k (\nabla_\theta S_k)^\top \Sigma^{-1} (\nabla_\theta S_k).
$$  

若 $\mathrm{rank}(F)=3$，则 $\theta=(\lambda,\beta,\alpha)$ 局部可识别。  

验证协议：对 $\alpha,\lambda,\beta$ 分别做 $\pm10\%$ 微扰，采样 $m$ 次估计 $\nabla_\theta S$ 并计算 $\det(F)$；阈值 $\det(F)>\delta_F>0$ 视为通过。  

---

## 推论 C（α→0 残余效应上界）

若 $f_\theta$ one-sided-Lipschitz，步长 $\eta \le 1/L$，$\alpha\in L^1([0,T])$ 且 $\alpha(t)\to0$，则生成轨迹与基线 CFM 收敛，满足  

$$
\mathbb{E}\|x_T - x_T^{\mathrm{CFM}}\| \le C \int_0^T \alpha(s)\,ds + O(\eta),
$$  

漂移/碰撞率 = $O(\alpha_{\max})+O(\eta)$。  

报告结果中应在图上同步标注 $\int_0^T \alpha(s)\,ds$ 的数值条带。




# 实验
-  **图 1（效率前沿）**：Top-1 < 2Å vs 采样/秒数/显存，报告 AUC。  

- **图 2（α 衰减收敛）**：两条 $\alpha(t)$ 日程与基线 CFM 的 RMSD 漂移/方差曲线。  

- **图 3（Fisher 满秩）**：±10% 扰动 $\alpha,\lambda,\beta \to S_1,S_2,S_3$ 的响应曲线（单调、互不混淆）。  

- **表 1**：Core-Only vs 主流生成式基线（Top-1/5, DCC, ECE/NLL）。  

## 能量解释验证
画 $\langle \nabla_x V_K, \Delta x\rangle$ 与 $\|\Delta x\|_W^2$ 的步级曲线，应满足“线性下降 + 二次正则”的最陡下降特征。  

---

## Fisher 满秩
对 ±10% 的 $\alpha,\lambda,\beta$ 扰动，绘制 $S_1,S_2,S_3$ 响应曲线并计算 $\det(F)$；  
以 $\det F > 0$ 为合格阈值。  

---

## $\alpha \to 0$ 残余效应
两条衰减日程（原始 / ×0.25）与基线 CFM 的 RMSD 漂移对比，并给出 $\int \alpha$ 的数值条件。  

---

## 离散等变误差
扫描 $h,\delta$，作 $\|\Delta x(g\cdot x) - \Delta x(x)\|$–分辨率曲线，验证命题的线性/二次收敛趋势。  给一张图

---

## 曲率屏障自适应
在 LB 平滑函数 $\sigma\in[2,10]$ 与网格分辨率下测 $\Delta x$ 方差，匹配上界趋势。给一张图


- ## 触发器机制

- **触发器 T1（单调失效）**：  
  若滑动窗口 200 步中 $\Pr[\Delta \mathbb{E}[H_+] > 0] > 0.3$，自动令 $\lambda \times 1.5 / \alpha_0 \times 0.5$，并记录事件。  

- **触发器 T2（离壁失效）**：  
  若 $\mathbb{E}[B(\phi)]$ 高于基线均值的 1.2×，则 $\beta \times 1.5$；若 Top-1 明显下降（>2%），则回退。  

- **触发器 T3（收益失效）**：  
  若 3 个 epoch 内 Top-1 无 ≥2% 相对增益，自动把 $\alpha_0$ 减半；再无增益则关闭引导，仅留耗散 + 屏障。  

触发器日志：T1/T2/T3 收敛与安全的自动调参事件合成一张“训练自适应曲线”，作为鲁棒性证据。


# 执行计划
- **时间线（周）**  
  - T0：数据与网格脚本（曲率估计）；基线 CFM 对接跑通  
  - T1：并入 ∇K 引导与 L_c；等变细化与碰撞过滤；日志/可视化  
  - T2：OOD 评测、校准、效率前沿与消融  
- **资源**：1×4090/24GB 或 1×A100/40GB，内存 ≤24GB，训练 ≤120h，评测 ≤60h  


# 风险与 PlanB
- **曲率估计噪声**：角亏/拉普拉斯-Beltrami 平滑；PlanB=以平均曲率 H 或到表面距离场替代 K  
- **∇K 引导过强致偏置**：α 退火与自适应裁剪；PlanB=仅在高 K 阈值区域启用引导  
- **L_c 难收敛**：分段线性代理目标与停止梯度技巧；PlanB=把 L_c 改为轨迹端点的 K 差分惩罚  

# 跨领域扩展
曲率-耗散 CFM 可迁移至蛋白-蛋白对接或 RNA 折叠路径生成，作为几何先验的稳态化动力学。


# 评测协议 
- **预算**：  
  - 样本：≤1.2e6 pose-steps（~18k复合物×口袋×≤64初始姿态）  
  - GPU 小时：≤180（1×4090/24GB 或 1×A100/40GB）  
  - 内存：≤24 GB  
- **目标**：  
  - Top-1 <2Å：相对提升 ≥8%  
  - 校准：ECE 下降 ≥0.03 且/或 NLL 下降 ≥0.20  
  - 稳定性：微扰保持率退化 ≤3%
  - 
# 备注
-  离散曲率估计实现细节
  ## 首选实现与公式来源（互相补）

1. **Meyer–Desbrun–Schröder–Barr**：  
   *三角网格上的离散微分算子*（含角缺陷高斯曲率、cotangent 拉普拉斯、平均曲率向量等；工程上最常用的配方）。  

2. **Rusinkiewicz**：  
   *曲率导数估计*（不规则网格、噪声高敏感性评估，给出主曲率/主方向的稳健实现思路，可与上文互证）。  

3. **Taubin**：  
   *曲率张量估计*（ICCV'95 经典做法，给出闭式正分解方案；在粗糙网格上可作交叉验证）。  

4. **评测与对比综述**：  
   *TriMesh 曲率估计方法系统比较*（Gatzke & Grimm），便于在附录量化你所用配方的稳定区间。  

---

## 可直接引用的实现要点（与上文一一对应）

- **顶点高斯曲率（角缺陷）**：  
  $K(v) = 2\pi - \sum_{f \ni v} \theta_f(v)$；  
  建议采用 *Voronoi/mixed area* 作为局部面积归一化，梯度用 **cot-Laplace** 实现。出处：Meyer 等。  

- **$\nabla K$ 的数值近似**：  
  以 cot-Laplace $\Delta \phi$ 的系数实现网格离散梯度/散度，。出处：Meyer 等。  

- **主曲率/主方向**：  
  按 Rusinkiewicz 的局部参数系有限差分/投影法稳健估计；可作为诊断曲线，验证 $K$ 场是否与形状直觉一致。

## CFM 与等变细化接口按现有代码库适配  
  ### Flow/CFM 核心论文与代码

- [Flow Matching 原始论文](https://arxiv.org/abs/2210.02747) （FM 框架总览、目标与实现细节）  
- [Flow Matching Guide & Code](https://arxiv.org/abs/2412.06264) （系统综述 + 设计抉择，含最新扩展，便于落地时对齐符号与训练细节）  
- [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) （官方 PyTorch 实现，含连续/离散 FM、API 文档与示例）  
- [TorchCFM](https://github.com/atong01/conditional-flow-matching) （Conditional Flow Matching 的轻量库，易于嵌入现有训练循环；支持 OT-CFM 等变体）  
- [Stochastic Interpolants](https://arxiv.org/abs/2303.08797) （从统一视角理解 FM/扩散；实现等式/损失的推导依据，便于自定义路径）  
- [FlowMM](https://github.com/facebookresearch/flowmm) （Riemannian Flow Matching 应用，在几何/材料域上的参考实现，可借鉴“几何引导项”的落点）  

---

### 等变建模/细化（E(3)/E(n)）代码

- [e3nn](https://github.com/e3nn/e3nn) （E(3) 等变算子与表示，产研通用底座；含文档与示例）  
- [EGNN](https://github.com/vgsatorras/egnn) （官方与社区版实现，便于快速起项目或做轻量等变细化头）  
- [DiffDock](https://github.com/gcorso/DiffDock) （等变/三维几何任务中的扩散-对接流水线，适合作为“几何细化接口”的参考骨架）  
- [TorchMD-NET](https://github.com/torchmd/torchmd-net) / [TorchMD](https://github.com/torchmd/torchmd) （分子几何与力场领域的等变网络与推理/MD 框架，便于嵌入“几何势”或正则）

