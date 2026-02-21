# perturbation_py 潜在贡献地图

## 1) 对研究者的贡献

- 提供轻量、可审计的一阶扰动求解器，不依赖 MATLAB/Octave。
- 将模型定义、导数计算、求解、仿真拆分为可组合 API，便于复现实证流程。
- 通过 Python 生态可直接接入数据管线、估计框架和可视化工具。

## 2) 对生态的贡献

- 在 Dynare 与 Python 宏观工具（Dolo/Sequence-Jacobian）之间补齐“经典 perturbation 求解”层。
- 提供一个更清晰的教学/研究基准实现，便于课程和方法复现。
- 可作为后续高阶求解、风险调整、pruning 的公共底座。

## 3) 对工程实践的贡献

- 测试优先的实现路径：每个数值模块都有可验证的最小模型。
- 错误信息面向建模者（BK 失败、矩阵奇异、维度不一致）。
- CLI demo 可直接用于快速 sanity check。

## 4) 与 Dynare 的关系边界

- 本项目目标是“方法移植 + Python 化工程实现”，不是 1:1 复制 Dynare 全功能。
- 首期范围限定在一阶扰动，优先保证结果稳定性和接口清晰度。
- 高阶特性（2/3 阶、pruning、occasionally binding constraints）作为后续增量。

## 5) 开源协作机会

- 新模型模板与 benchmark 用例贡献。
- 与 Dynare、Dolo 输出的自动对比脚本贡献。
- 文档双语化（中英）和教学示例扩展。
