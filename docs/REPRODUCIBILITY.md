# 环境复现、验证层级与故障排查

## 环境与固定资产

参考平台是原生 Ubuntu 22.04 x86_64、Python 3.10。第一轮完整 CI 使用 Ubuntu 22.04.5 / CPython 3.10.21；本地普通 Ubuntu 的 python3.10 安全补丁号可能不同，需保留 `doctor` 和 `pip freeze`。Windows 摄像头后端已有适配分支，但 Windows 不是本次主验收平台，不承诺跨平台完全相同的物理浮点结果。

`pyproject.toml` 固定直接依赖，`requirements.lock` 固定第一轮成功 CI 解析出的全部 Python 依赖版本。安装脚本优先使用 lock，再以 --no-deps --no-build-isolation 安装本项目。模型为官方版本化下载，SHA256：

```text
fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
```

这是可审计的版本锁，不是包含所有 wheel 哈希和 OS 镜像摘要的全平台字节级环境锁。跨 CPU/系统的动力学差异需要比较，而不是声称绝对比特一致。严谨重复实验时固定 git commit、同一机器、内参/掌宽、模型哈希、控制频率、种子及所有依赖。

## 在线安装与离线准备

```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
python -m gesture download-model
python -m gesture doctor
bash scripts/verify_all.sh
```

模型校验后，正常 live/demo/replay 不需要网络。只有显式 download-model、首次依赖安装以及可选 public-image smoke 会访问网络。

无法在目标电脑访问模型主机时，可在有合法网络访问的同平台电脑完成下载，再将原始 `models/hand_landmarker.task` 复制到目标电脑相同路径；运行 doctor 必须通过 SHA256 检查。不要关闭 SSL/校验或下载来源不明模型。不要在环境变量中配置与本项目无关的 AI API Key。

在同为 Ubuntu22/x86_64/Python3.10 的联网机器准备离线依赖：

```bash
python -m pip download --only-binary=:all: -r requirements.lock -d wheelhouse
python -m pip download pip==25.0.1 setuptools==75.8.0 wheel==0.45.1 -d wheelhouse
sha256sum wheelhouse/* > wheelhouse.sha256
```

复制源码、wheelhouse、校验文件与模型。目标机器应已安装 Python venv 和说明中的系统库；这些 apt 包不包含在 wheelhouse 中。目标机执行：

```bash
sha256sum -c wheelhouse.sha256
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links wheelhouse pip==25.0.1 setuptools==75.8.0 wheel==0.45.1
python -m pip install --no-index --find-links wheelhouse -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m gesture doctor
```

共享离线包前检查第三方库、模型、URDF/网格资产的重新分发许可。

## 验证层级

1. 单元测试：几何、尺度、坐标、滤波、状态机、数据校验、标定求解。它们不需要真实摄像头，不能替代摄像头验证。
2. 集成测试：PyBullet 真实物理步进下执行合成几何输入，通过三任务成功判据、命令回放误差、导出及重处理检查。合成输入直接构造关键点，不经过网络。
3. 模型检查：官方模型 SHA256 + CPU 实例加载 + 空白帧推理。另有 `python scripts/vision_fixture_smoke.py` 明确联网下载上游公开测试图片，检查单手/多手与视频通路；它仍不是笔记本实时摄像头试验。
4. GUI 检查：CI 的虚拟 X11 display 检查 OpenCV 窗口与循环可运行，不等于目标桌面驱动或摄像头权限已验证。
5. 人工现场验收：在目标笔记本完成 docs/EXPERIMENTS.md 的检查与测量。未开展的试次必须标为未测，不捏造性能。

本地运行 pytest 时，缺少 pybullet/mediapipe 会跳过相应模块。因此**只看到 pytest 退出码 0 不足以认为完整验证通过**；检查 skipped 数量，并运行 doctor。正式 Ubuntu CI 先安装全部依赖和模型，预期基础套件为 36 项通过、0 跳过。

## 常见错误

| 现象 | 处理 |
|---|---|
| 模型下载失败 | 确认 Google 官方地址网络可达；可按上文复制已验证模型；保留完整报错 |
| Model checksum mismatch | 不绕过校验；检查文件是否为 HTML 错误页、被代理修改或下载中断 |
| Cannot open camera | 关闭会议/浏览器占用，检查系统摄像头权限；尝试 `--camera 1`；不要以 root 长期运行应用 |
| 无桌面显示 | 使用原生桌面运行 live/calibrate；服务器使用 demo/replay 的 --headless |
| OpenCV Qt/xcb 报错 | 检查桌面会话和 libgl1/libglib2.0-0；不要同时安装多个 opencv-python/headless/contrib 包 |
| 画面中有手但不动 | 只保留一只手，观察拒绝原因，完整掌部朝向镜头，先 C 后 Space，窗口需要键盘焦点 |
| LOST / pose jump | 先固定相机、改善光照并慢移动；重新稳定后 Space 重锚定；不要直接提高所有阈值掩盖问题 |
| 深度与真实距离不符 | 检查 K、视频实际分辨率、数字裁剪和 5–17 掌宽；同人不同掌倾角也会有误差 |
| GPU/EGL warning | 本项目明确使用 CPU；在 doctor 完整通过时该 GPU 不可用提示不表示 CPU 推理失败 |
| Episode already exists | 改用新输出路径；不要覆盖旧试次造成数据混淆 |
| 机器人夹不到方块 | 先看合成抓放演示，默认不启用姿态映射，俯视对准，缓慢下降，检查是否因丢手或 IK 拒绝停止 |

`python -m gesture --debug <command> ...` 输出完整异常栈。反馈问题时提供 git commit、doctor 输出、summary.json、报错和操作步骤；不必上传有隐私的原始视频。
