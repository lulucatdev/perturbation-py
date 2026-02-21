# perturbation_py 方案设计

## 当前实现结构

- `src/perturbation_py/model.py`
  - `DSGEModel`：定义状态、控制、冲击、参数、稳态与用户函数接口。
- `src/perturbation_py/derivatives.py`
  - 数值中心差分导数，产出 `g_s/g_x/g_e/f_s/f_x/f_S/f_X`。
- `src/perturbation_py/solver.py`
  - 基于 `scipy.linalg.ordqz` 的一阶扰动求解，包含 BK 条件检查。
- `src/perturbation_py/solver_second_order.py`
  - 基于局部隐式控制映射的二阶张量近似（`ghxx/ghxu/ghuu/ghs2`）。
- `src/perturbation_py/solver_third_order.py`
  - 基于二阶导数差分构建三阶张量（`ghxxx/ghxxu/ghxuu/ghuuu`）。
- `src/perturbation_py/policy.py`
  - 统一策略对象，支持 1/2/3 阶控制规则求值。
- `src/perturbation_py/pruning.py`
  - 高阶策略的 pruning 仿真与 IRF。
- `src/perturbation_py/simulation.py`
  - 线性与高阶策略仿真接口。
- `src/perturbation_py/io/`
  - Dynare `.mod` 子集解析与参考结果 JSON 读写。
- `src/perturbation_py/benchmarks.py`
  - Dynare 对齐比较、fixture 运行与误差报告。
- `src/perturbation_py/cli.py`
  - `solve/steady-state/simulate/irf/parse-mod/demo` 完整命令流。

## 数据流

1. 用户提供模型函数与稳态点。
2. 导数模块在稳态附近做数值线性化。
3. 一阶求解模块构造广义特征值问题并恢复策略矩阵。
4. 高阶求解模块通过局部隐式映射构造二阶/三阶张量。
5. 策略模块统一计算控制规则，仿真模块生成路径与冲击响应。
6. 基准模块对接 Dynare 做 parity 报告。

## 与参考实现映射

- Dynare `dyn_first_order_solver.m` -> 本项目 `solve_first_order`
- Dolo `approximate_1st_order` -> 本项目 QZ 排序与策略恢复逻辑
- Dynare 高阶张量接口 -> 本项目 `solve_second_order`/`solve_third_order`
- DSGE_mod 模型族 -> 本项目 `tests/fixtures/dynare_parity_suite.py`

## 后续技术路线

- 自动微分后端（JAX）与符号导数后端，提升高阶精度与性能。
- 扩展 Dynare 语法覆盖（宏处理、varobs、估计块）。
- 引入官方 Dynare 运行环境容器并在 CI 中执行 parity integration。
