# PORTS：成功率优先与执行中恢复

## 本次日志能确认什么

用户提供的报告：43 / 116 成功，Micro SR=37.07%，平均任务内 adapter 切换为 0。52 个失败归到 `environment_not_satisfied`，21 个归到 `agent_finished_but_failed`。

`environment_not_satisfied` 在旧 reporter 中只是“benchmark 判定未成功，最后输出也不是结束动作”的汇总分类，不代表 emulator 初始化失败。日志确认本轮连接是 ADB=emulator-5556、gRPC=8555，这一对匹配。不能把本轮失败再次归因于之前修复的端口错配。

具体观察：

- AudioRecorderRecordAudioWithFileName：从第 3 步开始反复点击 index32 / index0，在命名下拉框和原界面之间循环，20 步结束，始终是 all02。
- BrowserDraw：先多次尝试 `File Manager`，之后重复点击 `task.html`，没有完成绘图；20 步结束，始终是 control。
- BrowserMaze：20 步结束，始终是 control。
- 日志中很多“Summary”实际上是 `{"action_type": ...}`，不是观察总结。这样的历史会误导后续动作。

关闭完整轨迹的 JSON 只剩聚合信息，不足以还原每次 UI 像素或证明所有失败的根因。这次针对已经可确认的执行/路由问题修改，不能宣称已消除所有 environment_not_satisfied，也不能保证 PORTS 必然超过对照方法。

## 路由改动

1. 默认 `[routing].objective="success_first"`。DP 首先最大化累计 `sum(log C)`；只有主目标完全相同时，才用技能奖励等打破平局。延迟/切换惩罚为 **0**，即使旧 TOML 还保留较高 `latency_weight` / `switch_weight` 也不会压过成功率。设 `objective="weighted"` 可显式恢复旧加权目标。
2. 每个 GUI 动作根据当前执行单元的模型—技能证据与当前 UI 原语重选 adapter，不再让一个 L3 技能把模型锁定一整项任务。动作和其总结仍使用同一个 adapter。
3. 当前 UI 的已知原语能力作为轻量上下文先验，不从 `control` / `crossapp` / `all` 等目录名臆造模型专长。
4. 检测重复的“界面＋动作”以及格式/执行错误。达到阈值后，下一步降低该 adapter 的优先级并尝试其他合格模型，暂时不注入当前技能；同技能再次造成循环，在当前 episode 剩余部分禁用它。不改 SQLite 中的技能生命周期或标定数据。
5. 技能阶段只由实际动作与可见状态的匹配作软推进，不再把调用次数当作已成功完成的步骤。到达计划末尾不等于任务已完成。
6. 提供来自当前 AndroidWorld action space 的合法 `open_app` 名称提示；这只是支持名称，不保证应用已安装。
7. 如果动作模型在总结阶段输出另一条动作，用当前步骤 UI 描述变化生成保守摘要，明确“已尝试”不等于“已成功”。真实模型/API 错误不会被这个保护伪装成成功。

运行时模型排序为：

```text
q_context = geometric_mean(当前上下文中有画像的原语能力)
S(m) = C(m,unit)^(1-w) * q_context^w * d^f(m)
```

默认 `w=0.25`，`d=0.35`，`f(m)` 是本 episode 中该模型被观察到循环/无效输出后的恢复次数（折扣最多累乘 4 次）。没有已知上下文能力时 `w=0`。无技能单步恢复用归一化的上下文原语先验，而不是把多个同时存在的特征误当成多步独立执行并相乘。

`S(m)` 是**用于恢复排序的 compatibility score，不是新实测 SR**。观察失败只是启发式，不会写回模型能力表。若各 adapter 证据完全相同且没有循环，仍可能合理地不切换；代码不会为“切换次数好看”而强制随机切换。

## 建议配置

在现有 TOML 的已有段中修改，不要重复添加同名表：

```toml
[routing]
objective = "success_first"
success_weight = 1.0
latency_weight = 0.0
switch_weight = 0.0
use_skill_estimates = true
runtime_rerouting = true
runtime_context_weight = 0.25
stall_threshold = 3
stall_model_discount = 0.35
```

旧 TOML 不新增这些字段也会启用新的默认策略；显式写出便于记录实验设置。既有 `minimum_capability`、技能状态准入规则仍有效，不会绕过真实禁用状态。

### 步数：命令写 30，为什么只有 20？

旧 AndroidWorld 会按 task complexity 先分配原生预算；现有兼容层默认使用 `min(原生预算, --max-steps)`。因此命令里的 30 是上限，很多任务仍只有 20。

若希望所有任务真正获得完整的 30 步，可在共用 TOML 中设置：

```toml
[evaluation]
fixed_step_budget = true
```

这是**所有使用这份配置的评测方法共用的设置**，不是仅给 PORTS 放宽预算。成功判定仍由原 benchmark verifier 提供；成功后提前停止的行为不变。

不要把“新 PORTS 固定 30 步”直接与“旧 standalone 原生 20 步”当作等预算增益。若要单独测路由改进，保留 `fixed_step_budget=false`；若改成 true，各对照也应重测。新超时记录单独归为 `step_budget_exhausted`，旧无足够信息的 checkpoint 不会被事后重标为成功。

## 服务器重测命令

以下沿用之前的 CUDA1 / 8003。TOML 的 emulator 仍应为你本次有效的 5556 / 8555；GPU / 模型端口与 emulator 端口互相独立。客户端工作区没有你的 `config.local_test1.toml`，需要你在服务器上修改实际配置。

**必须使用新的 output-dir**。旧目录中的已测失败会被 resume 跳过，而新路由协议也拒绝与旧结果混合。SQLite 可以继续使用原先的副本，不需要因为此次代码更新重新做完整标定。

先做小样本冒烟：

```bash
python -m src1 --config src1/config.local_test1.toml evaluate-pmtskill \
  --adapter-paths \
    /home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_control01 \
    /home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_crossapp01 \
    /home/zmz/Workspace/gui/src1/runtime/checkpoints/deprecated/adapter_all02 \
  --adapter-model-ids adapter_control01 adapter_crossapp01 adapter_all02 \
  --skill-database /home/zmz/Workspace/gui/src1/runtime/evaluations/compare_five_control/ports/skill_library.sqlite3 \
  --tasks AudioRecorderRecordAudioWithFileName BrowserDraw BrowserMaze ContactsAddContact \
  --routing-objective success_first --routing-latency-weight 0 --routing-switch-weight 0 \
  --routing-runtime-rerouting --routing-stall-threshold 3 \
  --combinations 1 --seed 42 --max-steps 30 \
  --cuda-visible-devices 1 --deploy-port 8003 --infer-backend vllm \
  --no-require-training-state --resume --no-store-trajectories --no-record-traces \
  --output-dir /home/zmz/Workspace/gui/src1/runtime/evaluations/ports_success_first_smoke
```

同指令加 `--dry-run` 可查看生效的 objective、runtime_rerouting、延时/切换权重和 fixed_step_budget，不启动模型。正式全量去掉 `--tasks ...`，并换成新的 `ports_success_first_full` 目录；不要在冒烟目录里改变任务集合。

运行中查看日志里的 `PORTS_ROUTE` / `PORTS_STALL`。报告保存 `routing_runtime`：动作级重选数、实际切换数、循环恢复数、无技能回退数、总结修复数。`--no-store-trajectories` 下这些统计与轻量 resume 仍会保留。

如需旧策略消融，使用新目录并加 `--routing-objective weighted --no-routing-runtime-rerouting`，恢复旧权重。基础设施不可用仍按不可用处理，不通过改分母、放松 task verifier 或使用任务答案提高 SR。

## 尚待服务器验证

本地自动测试验证路由边界与恢复逻辑，不代表实际 SR 已提高。日志还有 vLLM 对部分视觉层 LoRA 的忽略警告；需要对照实际 adapter 的 target modules / 权重确认是否确实丢失了训练后的视觉参数。不能仅凭这些警告认定它就是性能根因，本次没有擅自切换推理后端或修改 adapter 权重。
