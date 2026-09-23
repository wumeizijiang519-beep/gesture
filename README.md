# 基于 RGB 视觉的机械臂三维手势遥操作

**课程选题 13 · 主复现平台 Ubuntu 22.04 x86_64 / Python 3.10 · 笔记本 RGB 摄像头 · Franka Panda 仿真**

本项目包含可执行代码，不依赖云端 API、CUDA、ROS、深度相机或实体机械臂。输入可以是摄像头、固定相机拍摄的普通视频，或可重复的合成几何测试序列。所有运行都生成可校验的 episode，支持动力学回放、指标计算和同输入消融。

> **适用边界：**固定相机、单只完整可见的手、普通照明。单目三维位置是“相机内参 + 实测掌宽 + 神经网络手形先验”条件下的近似估计，不是无条件真实世界坐标。只控制仿真，不提供真实机械臂驱动。摄像头实测验收与自动化测试是两类证据，见 [验证说明](validation/STATUS.md)。

## 1. 实现内容

| 环节 | 实现 |
|---|---|
| RGB 输入 | OpenCV 摄像头；采集/推理独立线程；只保留最新帧；本地 CFR 视频 |
| 手部感知 | MediaPipe Tasks CPU，21 个二维关键点及局部三维手形；双手同时出现拒绝控制 |
| 三维恢复 | 实测掌宽定尺度、相机标定、7 点 SQPnP、正深度/重投影/可见性检查 |
| 重定向 | 相对三维位移、坐标轴映射、One Euro 滤波、离合重锚定、连续捏合夹爪 |
| 可选姿态 | `--orientation` 相对掌姿态映射，角度与角速度限制；默认关闭 |
| 仿真 | Panda 7 轴 + 双指夹爪，逆运动学、关节范围、末端目标碰撞检查、接触摩擦抓取 |
| 停止机制 | 丢手/过期数据/位姿突变停止、手动恢复、锁定停止、工作空间与目标速度约束 |
| 任务 | 到达目标、三维轨迹跟踪、方块抓取放置；明确成功条件 |
| 数据 | 原始关键点、三维估计、时间戳、控制事件、动作、机器人状态、环境版本、SHA256 |
| 复现 | 同动作动力学回放、状态展示回放、NPZ 导出、指标与独立 XYZ 图、同输入滤波消融 |

## 2. 安装与第一次运行

使用 **原生 Ubuntu 22.04 桌面系统**。第一次安装需要访问 Ubuntu 软件源、PyPI 和 Google 的官方模型下载地址；完成安装和模型下载后，日常推理离线运行。不要在系统 Python 中混装依赖。

```bash
git clone https://github.com/wumeizijiang519-beep/gesture.git
cd gesture
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

首先验证无摄像头闭环，画面左侧是合成手骨架，不是摄像头录像：

```bash
python -m gesture demo --task reach --seconds 9 --output runs/demo-reach --assert-success
```

无桌面的服务器使用：

```bash
python -m gesture demo --task pick-place --seconds 30 --headless \
  --output runs/demo-pick --assert-success
```

完整自动化验收：

```bash
bash scripts/verify_all.sh
```

**输出目录不得重复。**再次执行时使用 `runs/demo-reach-02` 等新目录；项目故意拒绝覆盖已有实验。`--assert-success` 会在任务未成功时返回非零退出码，不隐藏失败。

## 3. 标定相机与掌宽

### 相机内参

```bash
python -m gesture board --output calibration/checkerboard.svg
python -m gesture calibrate --camera 0 --output calibration/camera.json
```

将 SVG 按 **A4 横向、100% 原始比例**打印，关闭“适应页面”。用尺核验每格 **20 mm**。棋盘有 **9×6 内角点、10×7 格子**，不要混淆。摄像头面对打印板，采集至少 12 个清晰且不同位置、距离、倾角的视图：空格采集、Enter 求解、Esc 取消。过于相似视图会被拒绝；建议采集 18–25 张并覆盖画面边缘。

输出 `camera.json`、重投影报告和角点 NPZ。标定残差不能代替独立距离真值。固定笔记本屏幕角度，禁止运行途中移动相机或改变裁剪/数字变焦。分辨率宽高比改变需要重新标定。

### 掌宽尺度

测量食指掌指关节中心（关键点 5）至小指掌指关节中心（关键点 17）的直线距离，以米传入。例如测得 8 cm 使用 `--palm-width 0.080`。它不是整个手掌外轮廓宽，也不是腕到指尖长度。掌宽误差会引入尺度误差；不要直接把默认 8 cm 当作每个人的准确尺寸。

## 4. 摄像头遥操作

```bash
python -m gesture doctor --camera 0
python -m gesture live --camera 0 --task reach \
  --intrinsics calibration/camera.json --palm-width 0.080 \
  --seconds 180 --output runs/live-reach-01
```

首次只检查界面、还未打印棋盘时可省略 `--intrinsics`，但程序会明确标记 approximate K。此模式用于调通，不用于宣称准确的米制深度。

保持一只手完整可见，先在舒适中立位置停住。点击程序窗口使其获得键盘焦点，**C 接受中立位，再按空格接通离合**。然后缓慢上下、左右、前后移动手。不要用快速挥手进行第一次测试。

| 操作 | 功能 |
|---|---|
| `C` | 接受当前手的中立状态；不等于相机内参标定 |
| `Space` | 接通/松开离合；重新接通时以当前机器人位姿重锚定 |
| 拇指/食指捏合 | 连续闭合夹爪；张开两指使夹爪打开 |
| `X` | 锁定停止；`C` 和空格不能绕过 |
| `R` | 清除停止锁定，不移动机械臂回家；随后必须 `C`、空格 |
| `V` | 切换斜视/俯视仿真视角 |
| `Q` 或 `Esc` | 正常结束并保存日志 |

**位移映射针对原始、未镜像图像：**远离相机 → 机器人 +X；图像向右 → −Y；图像向上 → +Z。默认镜像只用于左侧自视画面，不修改几何。方向不直观时先用 `--no-mirror` 学习坐标映射。离合松开后可将手移回舒适位置，再接通继续；不会把这段复位动作传给机器人。

丢手、双手出现、深度异常、快速位姿跳变或过期数据会停止运动。恢复有效手部观测后仍须重新按空格。软件“锁定停止”不是实体机器人的安全认证急停。

## 5. 三个任务与录制

```bash
# 连续三维圆轨迹跟踪
python -m gesture live --task path --camera 0 --intrinsics calibration/camera.json \
  --palm-width 0.080 --seconds 180 --output runs/live-path-01

# 接触摩擦抓取，不使用物体绑定或瞬移
python -m gesture live --task pick-place --camera 0 --intrinsics calibration/camera.json \
  --palm-width 0.080 --seconds 180 --save-video --max-mb 512 --output runs/live-pick-01
```

抓取：张开夹爪，移动到橙色方块上方，下降到方块附近，捏合闭爪，缓慢抬高，移到绿色目标区，下降后张开夹爪，再抬起手臂。按 `V` 的俯视图帮助横向对齐。默认固定夹爪朝下，适合先完成平移与抓取；验证这些功能后再启用 `--orientation`。

episode 包含 `meta.json`、`steps.jsonl`、`summary.json`、`manifest.json` 和最终仿真预览。**只有显式 `--save-video` 才保存 RGB 视频**。录制容量达到预算时停止并保留数据，不删除旧实验。视频按可用帧顺序写入，精确时间关系应读取 JSONL 中的 `capture_time` 和 `video_frame_index`，不要用 AVI 播放时长代替真实采集时间。

## 6. 回放、导出与评测

```bash
# 在相同仿真配置下重新执行已记录的动作，并检查末端轨迹误差
python -m gesture replay --episode runs/live-reach-01 --headless

# 精确显示已记录状态；这是可视化，不是动力学验证
python -m gesture replay --episode runs/live-reach-01 --mode state

python -m gesture verify-data --episode runs/live-reach-01
python -m gesture export --episode runs/live-reach-01 --output reports/live-reach-01.npz
python -m gesture evaluate --episode runs/live-reach-01 --output reports/live-reach-01

# 同一条原始关键点序列、同一操作事件序列上的滤波消融
python scripts/compare_filters.py --episode runs/live-reach-01 --output reports/filter-comparison-01
```

NPZ 使用 `numpy.load(path, allow_pickle=False)`。缺失观测用 NaN 编码，需使用有效掩码；JSONL 用 null。评测输出包含有效观测比例、推理时间分位数、主机取帧至控制时间分位数、机器人对指令的跟踪 RMSE。合成几何才有已知三维真值；真实摄像头记录不会凭空生成三维准确率。

另有 `video --input clip.mp4 --auto-arm --headless --output runs/video-01` 将固定相机 CFR 视频重定向到仿真；`--intrinsics`、`--palm-width` 应对应该视频。**不支持移动视角补偿、VFR 时间戳恢复或从视频学习机器人策略**。普通 egocentric 视频不能直接当作固定相机输入。

## 7. 工程目录

```text
gesture/geometry.py      内参、坐标、尺度与 SQPnP
gesture/vision.py        MediaPipe、摄像头/视频输入
gesture/control.py       滤波、离合、姿态/夹爪与状态机
gesture/simulation.py    Panda 动力学、IK、任务定义
gesture/calibration.py   标定板与相机标定
gesture/data.py          记录、校验、NPZ
gesture/runtime.py       应用管线、回放、评测、消融
gesture/cli.py           全部命令行入口
tests/                   单元、动力学和模型测试
scripts/                 安装、完整验收、消融
docs/                    方法、实验、复现与文献关系
.github/workflows/       Ubuntu 22.04 自动化验证
```

## 8. 与参考文献的关系

**EgoInfinity** 的启发是将“感知 → 三维表示 → 质量检查/清洗 → 机器人重定向 → 可追溯数据”分层，而不是仅做一个关键点动画。本项目实现轻量固定相机版本，没有执行 EgoInfinity 的 MoGe/WiLoR/SAM/物体网格重建、接触重建或大规模视频挖掘。

**MINT** 的启发是明确区分相机、手部和世界空间并显式组合变换。本项目不调用或训练 MINT，不估计移动相机轨迹；固定相机假设下只输出相机空间近似手腕位置，再映射到机器人基座。详见 [方法](docs/ARCHITECTURE.md)、[实验协议](docs/EXPERIMENTS.md) 和 [引用与许可](docs/REFERENCES.md)。

程序适合作为本选题的完整工程基础。课程研究结论需要小组在真实摄像头上完成标定、受控测量和用户实验，不能将合成脚本成功率替代人工遥操作成功率。
