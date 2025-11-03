# 🧬 Project: LAbind-workflow

## 📘 1. 项目简介（Overview）
- [LABind: identifying protein binding ligandaware sites via learning interactions between ligand and protein （NC,2025）](https://www.nature.com/articles/s41467-025-62899-0)。  
- 论文做了什么：引入了配体的语义向量，利用蛋白质和配体的信息交互（graphtransformer）嵌入，对蛋白质结合位点进行预测。
- 解决什么问题：现有忽略配体信息；多配体同模型结合位点预测训练推理。

## 💡 3. 改进思路（Improvement Summary）
### 🧭 问题背景
- 原LABind框架仅通过配体全局语义向量和蛋白质残基向量（几何+语义），分别输入Graph_transformer中做cross_attention和self_attention，从而实现配体-蛋白信息交互；
- 模型框架
![LABind Architecture](./assets/architecture_overview.png)

- 

## 🧱 2. 版本进展（Version Timeline）

| 版本 | 日期 | 主要变化 | 状态 |
|------|------|----------|------|
| v0.1 | 2025-11-03| 实现论文代码复现，结果还在运行 | ✅ 完成 |
| v0.2 | 2025-11-05 | 引入 curvature-aware alignment 模块 | ⏳ 进行中 |
| v1.0 | 预计 2025-12 | 融合 uncertainty-aware weighting + 报告论文初稿 | 🧩 计划中 |

详见 [`versions/`](./versions/) 目录。

---



