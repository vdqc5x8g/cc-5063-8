# Surface Studio · 曲面贴附设计工具

解析最近点反求 + 多解分支连续选择的曲面贴附设计与贴花工具。纯 Python 标准库，
零第三方依赖；前端是 canvas 2D 软件 3D 视窗。

---

## 图形入口（曲面贴附主入口）

曲面贴附的**主要操作界面是图形编辑视窗**，不是命令行：

```bash
python3 -m surface_studio
# 打开 http://127.0.0.1:8765/  （默认自动打开浏览器）
# 可选：--host 0.0.0.0 --port 8791 --no-browser
```

在视窗里：

1. **真实曲面 / 原设计曲线 / 投影轨迹 / 贴附笔画**——主视窗，左键拖动旋转，滚轮缩放。
2. **贴附编辑**——右侧滑杆精确调整 `offset / lateral / scale / angle`；
   或在下方“图案编辑条”直接**拖动图案**。拖动后自动重跑真实投影，
   曲面笔画与局部畸变热力图随之更新（拖动图案即更新曲面图形与局部畸变）。
3. **分支策略**（顶栏下拉）——`自动（实测择优）/ 邻域延续 / 路径约束`，
   见下方“分支策略的实测择优”。
4. **添加分支锚点**——开启后在曲面上单击任意位置放置锚点；工具会吸附到最近的
   设计采样样本，并可在该样本的多解候选中选择你点击的那一支。
   `清除锚点` 移除全部锚点。
5. **求解诊断**——并排展示平面 / 弯曲 / 折返 / 球极奇点 / 折返分支 / 球面分支轨迹
   的数值诊断。
6. **旋转后导出两份图样**——生成 `pattern_A_*` 与 `pattern_B_*` 的 SVG/JSON/CSV/轨迹 CSV
   及 `report.json`（含每曲面 `branchSummary`）。

### 运行状态与异常可见性

- 视窗左上角的**分支跟踪浮层**实时显示当前策略、多解样本数、断点总数。
- **曲面图形异常**以红色 `×` 标在断点样本处；换支以紫色 `×`；锚定/已锁定样本为
  金色实心点；丢失后“最近可行点恢复”为青色圆环。
- **局部畸变异常**：右侧“局部面积畸变热力图”与贴花有限差分畸变表；
  分支浮层在内点正交残差或断点超阈值时给出红色异常行。
  （边界极小样本不参与内点正交判据，单独计为“边界极小”，不会被误报为求解异常。）
- **被放弃的候选与选择依据**：在曲面轨迹上**悬停**任一贴附样本，
  浮层列出该样本采用的分支、状态/原因，以及**全部被放弃的候选**
  （含其 `u, v, 距离` 与越界原因如 `BEYOND_U_MAX`）。右栏“分支跟踪”表按曲面列出
  候选数、换支/恢复/锁定计数、真实贴附误差，以及邻域延续 vs 路径约束的实测总代价。

---

## 多解分支连续选择

问题：设计曲线经过相近但不同的表面区域时，**最近的单点解不保证形成连续可用曲线**。
`surface_studio/branches.py` 做的是**路径级**求解，而不是固定用上一帧初值：

- 每个样本收集多个候选（全种子 + 上一批次候选的 warm-start 延续），
  保留可行候选集合，而不是只留一个最近点。
- 再在**整条曲线**上做选择：`neighbor`（邻域参数延续贪心）与
  `path`（Viterbi 动态规划，emission=距离，transition=参数跳变 + 切换惩罚）。
- `auto` 同时跑两种策略，按实测总代价择优，并在 `strategyComparison` 中记录两种策略的
  总代价与所选路线，使“据实测选定”可审计。
- 明确**距离与连续性权衡**：`continuity_weight / distance_weight / switch_penalty`
  显式作用于代价；不把所有跳变用后期平滑掩盖——真实物理断开会报告
  `PHYSICAL_DISCONTINUITY_BEYOND_TOLERANCE` 并截断区间，而不是空中补线。

### 沿路径锁定与恢复

- 人工锚点作为**硬约束**，锚定后沿路径向前**锁定**同一分支；
  锚点之间的样本按锁定分支推进。
- 过接缝/折返导致锁定分支丢失时，回到**最近可行点**继续（状态 `recovered`，
  原因 `NEAREST_FEASIBLE_AFTER_LOCKED_BRANCH_LOST`）。
- 锚点若迫使跨越不连通区域，报告 **`anchor-not-continuable`**（不可连续），
  不生成连接线。

### 周期参数与接缝

- 周期参数按最短路径做 unwrap，避免穿过 `2π` 接缝时的伪跳变。
- 物理间隙阈值随设计曲线采样间距自适应，避免稀疏采样被误判为断点。

---

## 命令行（仅辅助）

命令行只用于无头回归与导出，**不替代图形入口**：

```bash
# 贴附多解回归例（折返曲面 / 两个等距候选 / 球面接缝极 / 沿边界移动 / 锚点）
python3 -m surface_studio.branch_regression
python3 -m surface_studio.branch_regression --json /tmp/branch.json

# 无头演示 + 导出两份图样与 report.json
python3 -m surface_studio.demo
```

`branch_regression` 退出码为 0 当且仅当全部用例通过，并打印
`Graphical entry point: python3 -m surface_studio`。

---

## 测试

```bash
python3 -m unittest discover -s tests -v
```

- `tests/test_branches.py`——多解分支回归：
  折返曲面（每样本两候选、最近点会翻面但跟踪不翻、断点被如实报告而非平滑、
  区间在断点处闭合、距离/连续性权重权衡）、
  两个等距候选（等距永不换支、候选确实等距）、
  球面接缝/极点（跨极保持连续经线、区间与分支记录）、
  锚点（锁定+最近可行点恢复、冲突锚点报告不可连续、body 锚点不泄漏到 spout）、
  参数连续性（周期参数走短边、沿边界扫掠保持身份并报告退出）。
- `tests/test_solver.py`、`tests/test_workflow.py`——单点求解器、曲面身份、
  转接线换片、接缝、旋转不变性。

## 目录

```
surface_studio/
  branches.py       多解候选收集 + 路径级分支跟踪（DP/贪心 + 锚点锁定）
  branch_regression.py  多解回归 CLI（辅助）
  projection.py     曲面身份行走，消费每曲面分支轨迹
  surfaces.py       参数曲面与解析一阶导（含折返、球、壶嘴）
  scenario.py       场景与多解回归用例（折返/等距/边界）
  server.py         HTTP 服务与 JSON 载荷（branchStrategy / branchAnchors）
  web/              canvas 视窗（分支浮层、锚点、分支跟踪报告）
  export.py         SVG/JSON/CSV/轨迹导出（report.json 含 branchSummary）
```
