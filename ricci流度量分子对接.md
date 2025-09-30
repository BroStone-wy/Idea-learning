
## 研究角度1：从度量层面把ricci曲率流融入图模型，改变随机游走/最短路等全局行为，从几何原则改善“瓶颈/通道”可达性 Ricci-Flow Metric Learning for Protein–Ligand Docking
- 有区别于用离散ORC/FRC无论是做特征工程或是网络层中局部加权，把Ricci-流后度量接入docking 的统一端到端训练目标，也可以与生成式 docking 的天然耦合：DiffDock 将对接视为平移×旋转×扭转的非欧流形生成；把曲率引入扩散/流匹配（如噪声/漂移的几何整形）可在同一个训练目标里统一“几何先验 + 生成”，避免后置重排序等行为。
- 创新点：新的曲率度量空间用于分子对接过程，而不再是局部加权；作为几何先验可训练目标参与到生成式对接模型。
- 对比参考文献：
  - Curvature Graph Neural Network/Curvature-enhanced GCN/CurvAGN ricci曲率影响局部消息传递权重（如邻居权重加权、attention 权重、边选择）
  - Ollivier Ricci-flow on weighted graphs 带权图上研究 Ollivier-Ricci 流的存在、唯一性、正规化形式等。
  - Network Alignment by Discrete Ollivier-Ricci Flow 用“Ricci flow metric”来定义图节点间新距离，用于图对齐
  - Graph Ricci Flow and Applications in Network Analysis 探讨 Ricci flow metric 在社区检测、距离量化上的应用

- 解决问题：
  - 度量级别重构，统一几何尺度，避免层间矛盾 / 信息冲突，模型可解释性更强。
  - 重构度量是几何-概率一体化操作，更贴近曲率作为几何特性本质。
  - 在不同体系 /不同受体-配体对，度量变换可学习实现自适应调整，提升泛化性。

- 现有落地基础：
 - 其他加权方式ricci曲率加权方法架构，作为baseline；
 - 通用图结构学习的ricci流度量算法，为改进适配于分子表征/对接/评分提供基础。

