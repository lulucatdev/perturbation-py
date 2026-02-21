# perturbation_py 可行性分析

## 结论

`perturbation_py` 作为一个开源 Python package 是可行的，且首期聚焦一阶扰动（first-order perturbation）可以快速形成稳定 MVP。核心数学与数值模块在 NumPy/SciPy 生态中可直接实现，工程风险主要在模型接口设计、稳态一致性校验、以及与 Dynare 在边界案例下的结果对齐。

## 可行性依据

- 算法可迁移性高：Dynare 一阶求解本质依赖雅可比矩阵和广义 Schur (QZ) 分解，SciPy 已提供成熟实现。
- 参考实现充分：本地 `references/dynare/`、`references/dolo/` 提供了 C++/MATLAB/Python 三种层面的对照。
- 工程依赖轻：首期仅需 `numpy`、`scipy`、`pytest`、`typer`，无编译型依赖。
- 验证路径明确：可使用线性可解析模型和标准 RBC 小模型做回归测试。

## 主要风险

- Blanchard-Kahn 判别在近单位根情形下数值脆弱。
- 自动稳态求解与用户自定义稳态并存时，容易出现“稳态不一致但仍可线性化”的隐性错误。
- Dynare 的变量时序约定（lag/lead incidence）与 Python 直觉接口存在语义差异，需在文档中明确。

## 风险缓释

- 在 solver 中显式报告 BK 条件失败原因与特征根计数。
- 强制在稳态点验证 `transition` 与 `arbitrage` 维度与残差形状。
- 用固定基准模型（标量模型 + RBC）做 CI 回归，先做一阶一致性，再推进二阶。

## 分阶段可交付

- Phase 1（已实现）：一阶扰动、QZ 求解、IRF/仿真、CLI demo、测试。
- Phase 2：二阶张量、pruning、更完整的报告结构。
- Phase 3：`.mod` 解析适配层、与 Dynare 输出对比工具、性能优化（Numba/JAX）。
