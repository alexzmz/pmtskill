# AndroidWorld 任务录制：步骤视频与控件视频

在服务器仓库根目录，激活平时运行 AndroidWorld 和 vLLM 的 Python 环境。
需要 Pillow、可执行的 ADB 和支持 `libx264` 的 FFmpeg。录制器不负责启动 emulator，
也不配置 gRPC；复用评测 TOML 中 `[android_world]` 的设置。

## 示例：OsmAndTrack / control01 / GPU 0

1. 启动并检查现有的 AndroidWorld emulator：`adb -s emulator-5554 get-state` 应返回 `device`。
2. 检查 `config.example.toml` 的 `evaluation.config`、adapter 路径及模型服务端口。
   样例引用仓库的评测配置，使用 `5554/8554`（console/gRPC）、`8002`（vLLM）、物理 GPU 0。
   如果你实际使用别的评测 TOML，复制录像 TOML 并把 `evaluation.config` 指向它。
   TOML 中的相对路径都从录像 TOML 所在目录解析。
3. 默认先跑 standalone，仅执行 `OsmAndTrack` 一次：

```bash
python -m src_video --config src_video/config.example.toml --dry-run
python -m src_video --config src_video/config.example.toml
```

等前一条结束，再用同一台模拟器跑 PORTS：

```bash
python -m src_video --config src_video/config.example.toml --method ports
```

也可执行 `bash src_video/run_osmand_track.sh`，或传入 `--method ports`。
单 adapter 的 PORTS 示例仍运行 PORTS 的规划与技能逻辑，但不会在多个 adapter 间切换。
要录多模型 PORTS，填写对应的 `adapter_paths`、`adapter_model_ids`；
技能库与规划模型可通过录像 TOML 覆盖，省略则继承正常评测设置。
不要同时在 `5554` 上跑两个评测。录像会保留正常评测轨迹，但不会把测试轨迹写入技能库。

## 输出与含义

默认输出到 `src1/runtime/videos/OsmAndTrack_<method>_<run-id>/`，每次实际尝试都有独立的 `episode_NNN/`：

- `steps.mp4`：真实屏幕采样时间线；在对应时间插入 M3A 决策前的原始轨迹截图，用黄色边框标记 **Step 1、Step 2…**（原生轨迹 `step_number + 1`）。不拿画面相似的截图冒充轨迹原图。
- `controls.mp4`：同一时间线，叠加模型最近一次观察中可操作控件的青色框及原始 SoM 编号。画面仍与观察相近时延续标注，检测到画面变化而尚无新观察时暂停标注；因此它反映的是**模型最近的感知**，不是独立检测器的逐帧真值。
- `capture.json`、`videos.json`：保存采样时间、轨迹截图来源、控件范围、步骤匹配统计和评测判定。任务失败与基础设施异常（`success: null`）分开。
- `evaluation/`、`cli_logs/`：原有评测结果和日志。原始 PNG 保存在 episode 目录，便于检查和重编码：

```bash
python -m src_video --render-existing /path/to/run/episode_001
```

录制从任务初始化完成后开始，在任务清理前截取终态，编码在评测结束后进行。
默认 ADB 每秒约采 4 帧，输出 12 fps；补帧仅保持最近的真实图像，不生成中间画面。
较短的动画可能落在采样间隙。录制本身有 ADB 和磁盘开销，建议不要用开着录像的运行报告论文的正式 latency。
截图可能包含任务中的敏感内容，分享前请检查。

## PORTS 路由原理动画（不需要模拟器）

```bash
python -m src_video.ports_workflow --output deliverables/ports-recipe-workflow.mp4
```

脚本用 `RecipeAddSingleRecipe` 作为教学示例，展示任务输入、小模型草案、原语分解、
L1/L2/L3 技能匹配、DAG 边代价、带模型状态的动态规划以及最优路径回溯。
配方内容、技能和代价为演示数据；路径由脚本计算，不是实测路由结果。
需要 Pillow；MP4 还需要支持 libx264 的 FFmpeg（可用 `--ffmpeg /path/to/ffmpeg` 指定）。
没有 FFmpeg 时，将输出扩展名改为 `.avi` 即可生成同内容的 Motion JPEG 视频。
可用 `--preview deliverables/ports-recipe-preview.png` 仅生成六幕预览。
