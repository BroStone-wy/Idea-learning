# 🧬 Project: LAbind-workflow

## 📘 1. 项目简介（Overview）
- [LABind: identifying protein binding ligandaware sites via learning interactions between ligand and protein （NC,2025）](https://www.nature.com/articles/s41467-025-62899-0)。  
- 论文做了什么：引入了配体的语义向量，利用蛋白质和配体的信息交互（graphtransformer）嵌入，对蛋白质结合位点进行预测。
- 解决什么问题：现有忽略配体信息；多配体同模型结合位点预测训练推理。

## 💡 3. 改进思路（Improvement Summary）
### 🧭 问题背景
- 原LABind框架仅通过配体全局语义向量和蛋白质残基向量（几何+语义），分别输入Graph_transformer中做cross_attention和self_attention，从而实现配体-蛋白信息交互；
- 模型框架
<img width="544" height="541" alt="image" src="https://github.com/user-attachments/assets/6dcb8ec4-5da6-465a-9fcc-ee5765ff2168" />

### 🧠 切入思路
1.**语义分解（把配体全局向量拆成多子语义）**
- 模型表现：
  - 蛋白质残基特征（几何+序列）经过 cross-attention 被全局配体语义调制；
  - 模型学到的是残基特征与配体类别的统计相关，“不知道为什么结合，只知道这类配体常常和哪类残基配对”；
  - 缺少建模结构功能相关性，特征相互作用表达的注意力解释性差，对于未见过配体的结合方式泛化性差。
- 改进思路：
  - 配体语义局部功能化；局部+局部信息交互，具体详见[`versions/v1.0/`](./versions/v1.0/) 目录。

## 🧱 2. 版本进展（Version Timeline）

| 版本 | 日期 | 主要变化 | 状态 |
|------|------|----------|------|
| v0.1 | 2025-11-03至2025-11-04| 实现论文代码复现，结果指标和论文基本差不多，部分2%可接受 | ⏳ 进行中 |
| v1.0 | 2025-11-04 | 设计并引入配体语义的可微聚类模块；考虑表面特征是否引入 | 🧩 计划中 |
<!-- ✅ 完成  ⏳ 进行中 🧩 计划中 -->



---



