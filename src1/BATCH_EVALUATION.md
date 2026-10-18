# 五方法批量评测

四个 baseline **只用 adapter_control01**；PORTS 使用 adapter_control01、
adapter_crossapp01、deprecated/adapter_all02 三个 adapter。三个 adapter 必须兼容
同一基座，脚本会提前检查，不强行覆盖成错误的基座。模型 ID 与离线 profile 保持一致。

```bash
cd /home/zmz/Workspace/gui

python -m src1.batch_evaluate \
  --config src1/config.local_control.toml \
  --skill-database /home/zmz/Workspace/gui/src1/runtime/skill_library_sqlite/v2/skill_library.sqlite3 \
  --cuda-visible-devices 3 --deploy-port 8005 \
  --combinations 2 --seed 42 --max-steps 30 \
  --output-dir /home/zmz/Workspace/gui/src1/runtime/evaluations/compare_five_control
```

先追加 `--dry-run` 打印全部命令，不读取权重、启动 GPU、连接模拟器或写文件。
实际运行还会调用五个入口的 dry-run 做 checkpoint/命令预检，再开始第一项。
默认测试全部任务；建议先用 `--tasks ContactsAddContact` 跑一轮冒烟测试。
也支持 `--task-count 10` 固定抽样，任务列表冻结在 batch_manifest.json。

运行顺序：standalone → simple-skill → SKVM → V-Droid-Qwen → PORTS。
共用同一个 TOML、GPU、端口、任务列表、组合数、seed、步数上限，每项结束后释放
服务再开始下一项。默认不保存完整轨迹；保留轻量 checkpoint、报告、日志和 SKVM
编译/能力评估产物。要保存完整轨迹，加 `--store-trajectories`。

默认开启 resume，再次运行同一命令即可跳过已完成的方法，以及中断方法内已完成的
任务实例。正常任务失败也算已完成，不会只重测失败任务来抬高 SR。权重或配置变化
需要新的 output-dir。技能库使用首次运行时的冻结快照，不跟随源库更新。
默认遇到失败停止；`--continue-on-error` 可继续后续方法。错误/不完整结果显示 N/A，
不伪装成 SR=0。Ctrl+C 先通知子进程退出，让部署上下文释放服务。

## 输出

```text
compare_five_control/
  batch_manifest.json
  batch_progress.json
  commands.json / commands.sh
  preflight.log
  comparison.md / comparison.csv / comparison.json
  inputs/skill_library.sqlite3
  standalone/              # report.md、summary.json、checkpoints、console.log
  simple_skill/            # 独立技能库副本
  skvm/                    # 另有 skvm_report.md/json、编译产物
  vdroid/                  # 明确标注 Qwen 改造版
  ports/                   # 三 adapter 池、独立技能库副本
```

汇总包含 Micro/Macro SR、三类能力、相对 standalone 的 Micro SR 百分点变化，
以及步数、耗时、模型切换。原始技能库不被修改。默认只允许 active polished；
`--include-candidates` 可额外放开 candidate，不自动晋升。
Simple-skill 与 SKVM 使用相同可用技能集合（active polished + 现有检索逻辑认可的 raw）；
PORTS 使用现有分层规划/路由，不改变生命周期或离线估算记录。

## 新增 baseline 入口与依赖

- `evaluate-skvm`：复用旧 runner 的真实 SKVM profiler/AOT/runtime，但使用和
  standalone 一样的 src1 M3A、报告和恢复协议。默认 profiler + p1/p1p3 编译，
  每次请求选择适配变体，无 PORTS 模型切换，多模态截图传给模型。缓存按实际权重和
  技能内容隔离，不把 PORTS prior 当作 TCP。需要 `skvm` 可执行文件，或 Bun 与已安装
  依赖的 `libs/skvm`。首次准备可能较慢。`--skvm-prepare reuse` 要求完整 TCP，默认允许缺失变体明确回退；
  `--skvm-command /path/to/skvm` 指定程序。
- `evaluate-vdroid-qwen`：**改造版，不是原论文结果**。保留原生 UI 候选枚举、动作
  补全、工作记忆和搜索，固定 Qwen adapter 替代专用 Llama verifier 和 GPT。
  分数是 `P(Yes)/(P(Yes)+P(No))`，不是原版训练的 value head。
  服务需支持 chat `logprobs=true, top_logprobs=20`，且 Yes/No 同时有有效概率；
  不支持在 warm-up 报错，不伪造分数。请求关闭 thinking 以取得直接 Yes/No token。
  需要 V-Droid 源码、Ray 等依赖，但不下载 Llama/verifier、不调用外部教师 API。
  `--vdroid-root /path/to/v-droid` 覆盖源码目录。

原 `evaluate-vdroid` 原生 Llama 入口不变。新入口均读取主配置 TOML。

### SKVM 工具调用协议修复

ms-swift 可能同时返回有效的 `tool_calls` 和 `finish_reason="stop"`，旧 SKVM
会把它当作 `end_turn`，在执行工具之前退出，导致“1 step、文件不存在、L0”。
现有 Python 启动器会自动启动仅监听本机的兼容桥，将这类有效调用的结束标记
修正为 `tool_calls`；不从文本猜测工具，不修改参数，不改动截断或过滤响应。
因此继续使用 `/root/.local/bin/skvm` 二进制也可以生效，无需先重新编译。
必须同时同步根目录 `src/` 和 `src1/`，包含 `src/skvm_artifacts.py`、`src/skvm_stability.py`，不能只同步 `src1`。

旧 TCP/AOT 缓存保留原样，新结果自动放在原缓存根目录的 `openai-tool-calls-v2/`
子目录，避免把旧的错误 L0 画像继续复用。使用默认 `--skvm-prepare auto` 即可重新准备，
不必把批次参数改成 force。当前日志中应出现 `SkVM tool-call compatibility bridge`，
准备完成后 `skvm/protocol-compat.json` 记录被修正的响应数量。
正式评测阶段仍直接连接原模型服务，不经过这个桥。
如果旧 SKVM 已经产生正式 episode，批量脚本会自动改用
`skvm/openai-tool-calls-v2-bounded-tool-evidence-v1-valid-aot-or-original-v1/`
保存新协议的重测结果，不改动旧文件，也不重测其他已完成方法。
只有 profiling/AOT 阶段中断的批次继续使用原来的 `skvm/` 目录。
单独使用 `evaluate-skvm` 时，旧 episode 目录会被明确拒绝，需要指定新输出目录。

仓库 `libs/skvm` 子模块中的 provider 也已做同样修复，使用源码或重新构建时生效。

### AOT 格式与上下文预算修复

兼容已发布二进制的顶层 `scr/gaps/pass3.dag`，以及源码版的
`artifacts/passRuns`。转换仅发生在内存中，旧 JSON 不被改写，也不伪造
旧版不存在的 pass 成功记录；`variants.csv` 中用 `plan_schema` 和
`pass_telemetry_available` 区分。无 gap 时变体与原文相同是合法的编译结果。

`--skvm-prepare auto` 会检查 `SKILL.md`、编译计划、SCR、guard 和新版 pass 状态。
在 `--skvm-failure-policy strict` 下要求所有指定变体合格；默认的 `fallback` 策略见下文。
编译器预先复制但未编译成功的原始 `SKILL.md` 不算成功的 AOT 缓存命中。
重编译前的旧计划及技能成对备份到缓存中的 `kernel-backups/`，防止失败后用旧计划认证原文。
不必删除缓存、不必改 TOML、不必换 SkVM 二进制；保持原批量命令和 `--resume` 即可。

准备阶段的本机转发层采用 `bounded-tool-evidence-v1`：输出上限不超过
`min(8192, context_window / 4)`，超长工具结果和编译器附带的 `_profiling/` 日志做明确标注的摘录。
工具输出共享最多 12000 字符的预算；原任务指令、原技能、其他依赖文件及工具参数不被截断。
完整数据仍在磁盘上，模型可通过工具分段读取或执行脚本处理。
上下文窗口取 `--max-model-len`，未指定时使用 TOML 的 `training_evaluation.max_model_len`。
这不是精确 token 计数；若服务仍返回上下文错误，最多再尝试两次更小摘录，仍失败则明确报
`SKVM_CONTEXT_WINDOW_EXCEEDED`，不进入正式测试，也不把它当成真实 SR=0。

旧 TCP 中带有上下文 HTTP400 的 primitive 会被定向重测（保留其他 primitive），
依赖这些 primitive 的 AOT 会重新编译；修复清单及每个变体的 TCP 指纹持久保存，中断后可续跑。
这是纠正了标定依据，相关变体需要更新，不仅仅是解析旧 JSON。
`--skvm-prepare reuse` 不会修复这类 TCP，而会拒绝使用并提示切换 `auto`。
`skvm/protocol-compat.json` 记录窗口、策略、摘录次数、输出限额、重试和未解决错误数；
正式报告 metadata 中也保存准备统计。上下文策略更改前的正式 episode 不与新结果混跑，
批量脚本自动隔离到上述新子目录；本次若仅在 profiling/AOT 中断，则保留原目录续跑。

### SKVM 稳定 baseline：单个变体失败不阻塞整场测试

`evaluate-skvm` 和批量脚本现在默认使用 `--skvm-failure-policy fallback`。
现有批次只在准备阶段失败时，可以直接用原命令、原目录、`--resume` 重跑，
不用删缓存、不用改 TOML、不用修改或重新编译 `libs/skvm`。

每个技能的固定准备规则如下：

1. 优先复用已通过校验的 `p1p3`；如果它不合格但已有合格 `p1`，立即复用 `p1`，不再花一轮时间升级。
2. 没有合格缓存时，先编译 `p1p3`，失败的技能做有限重试，再尝试 `p1`。
   `--skvm-compile-retries 1` 表示每个变体额外重试一次（默认值）；可设置 0–3。
   已成功的技能不会因同一批次其他技能失败而重编。
3. 所有指定变体均不合格时，保留原始技能，明确标记为 original/unadapted fallback。
   不修改 `guardPassed`，不忽略验证错误，不删除 AndroidWorld 测试任务，也不修改官方任务得分。
4. 配置错误、缺失 API key、模型服务不可用或损坏的 TCP 仍然报错；
   不能把整套框架不可用伪装成正常 SKVM 测试或 SR=0。

这是一种明确声明的容错改造版 baseline，并不承诺一定完成（例如模拟器或 GPU 真正故障时仍需恢复）。
若所有技能均回退原文，任务 SR 仍按真实结果报告，但 `skvm_report` 不会标记为有效的 SKVM 适配效果对比。
原始技能回退的请求不计入 `skvm_adapted_requests`；混合请求同时报告请求级和技能选择级覆盖率。

开始正式测试前，选择的技能正文、计划和 TCP 会冻结在本轮目录，
`skvm/preparation.json` 记录每个技能的选用版本、失败原因、guard 失败数、编译尝试数和 AOT 覆盖率。
技能文件位于 `skvm/prepared-skills/<generation>/`。自动续跑只复用这些快照，
不会因为共享缓存后来多出一个更好的变体而改变后半程条件。
已有正式 episode 时不允许 `--skvm-prepare force` 改写技能选择；需要新输出目录。

`skvm/runtime-stats.json` 只保存聚合计数，即使 `--no-store-trajectories` 也会保存，
用于累计中断前后的适配/回退次数；不含逐步动作、截图或请求正文。
`summary.json` metadata、`skvm_report.json/md` 和批量 `comparison.json/csv/md`
均注明适配覆盖率/原文回退情况。

若需要恢复原来的严格模式，添加 `--skvm-failure-policy strict`。
独立旧入口 `src/task_runner_skvm.py` 保持 strict 默认值，需用下划线参数
`--skvm_failure_policy fallback` 显式开启新策略。

## 解读限制

PORTS 有三个 adapter，其他方法只有一个，增益包含模型池资源的贡献，不是等模型
数量的纯算法消融。V-Droid 使用独立 AndroidWorld fork，同名任务/seed 不保证两个
版本的参数实现完全相同，需要检查 checkpoint 中的 goal/task_seed。
不删掉 baseline 成功任务，不把基础设施故障计成任务失败。
本地单元测试不等于 Linux 真实 GPU/模拟器端到端验证。
