# 工程交付与源码包复现

主复现平台为 Ubuntu 22.04 x86_64 / Python 3.10。运行方式、相机标定和按键见根目录 README；方法、实验、已执行测试分别见 ARCHITECTURE.md、EXPERIMENTS.md 和 ../validation/STATUS.md。

## 交付物的区别

- **源码仓库 / 源码 ZIP**：完整程序、固定依赖清单、安装与验收脚本、单元和集成测试、中文文档。不含第三方 Python 安装包或模型权重，首次安装需联网。
- **ubuntu22-validation**：某一次 CI 的 JUnit、三任务记录、导出数据、数值指标与图。它是自动化验证证据，不是真人遥操作数据。
- **本地 runs/**：小组实际操作产生的实验。不要直接提交 RGB 录像或个人操作记录到公开仓库。

## 使用固定版本

```bash
git clone https://github.com/wumeizijiang519-beep/gesture.git
cd gesture
# 如需严格对应某次验收，先 git checkout <该次运行的完整 commit SHA>
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
bash scripts/verify_all.sh
```

课程实验开始后记录完整 commit SHA，不要把不同版本结果混入同一张表。`requirements.lock` 固定 Python 依赖版本；官方手部模型由程序进行 SHA256 校验。

## 无 Git 工作目录的源码包

每次成功执行更新后的 Reproducibility 工作流，会生成 `gesture-source-<完整提交 SHA>` 附件，其中是 `gesture-source.zip` 和同名 `.sha256` 校验文件。源码 ZIP 包含顶层 `gesture/` 目录和 `SOURCE_REVISION.txt`。

下载并解开 GitHub 附件外层 ZIP 后，在校验文件所在目录运行：

```bash
sha256sum -c gesture-source.zip.sha256
python3 -m zipfile -e gesture-source.zip ./project
cd project/gesture
cat SOURCE_REVISION.txt
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
python -m gesture demo --task reach --seconds 9 --headless \
  --output runs/archive-first-run --assert-success
```

`SOURCE_REVISION.txt` 是生成源码包时的提交标识。在没有 `.git` 的安装中，episode 的 `git_revision` 可能显示 unknown；归档实验时一并保存这个文件，不能补写为未经核验的 Git 提交。源码 ZIP 不是完全离线安装包：仍需按安装脚本下载依赖和已校验的官方模型。

CI 除了在仓库目录运行全套测试，还会把源码包解压到临时目录，运行 9 秒到达任务并断言成功。这能发现源码包漏文件或入口依赖当前仓库的部分问题，但不能替代你的真实摄像头验收。

## 自己生成可分享的源码包

完成修改并提交到 Git 后：

```bash
python scripts/package_source.py --output dist/gesture-source.zip
(cd dist && sha256sum -c gesture-source.zip.sha256)
```

脚本仅打包当前提交的受跟踪文件，不加入未受跟踪的录像、模型缓存、虚拟环境或临时结果。它拒绝未提交的受跟踪修改、已有目标 ZIP 或已有校验文件；每次使用新文件名或明确处理旧交付物。打包脚本本身有 8 项自动化测试，覆盖内容/提交/校验、一致字节输出、拒绝覆盖、拒绝脏工作区等行为。

## 笔记本现场验收顺序

先运行合成到达任务；然后 `python -m gesture doctor --camera 0` 检查设备。完成相机内参标定、实测掌宽，在摄像头窗口按 C、空格后再缓慢操作。依次检查上下/左右/前后位移、捏合开闭、松开离合、丢手停止和 X 锁定停止。最后录制三项任务、回放、导出并计算指标。

不要将“44 个测试通过”或“3 条合成任务通过”报告为真人遥操作成功率，也不要将机器人跟踪指令的毫米级误差报告为摄像头三维估计误差。最终测试数量和状态以所选 commit 的 CI 原始记录为准。
