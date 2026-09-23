# 最新交付验收：2026-09-23

本文件是验收报告，不改变程序。已验收的代码版本为 `b494dd46fe32eb1f74c3f162aed65a1b64d37d2f`。此前的 36 项测试记录保留在 STATUS.md；本次新增 8 项源码交付测试，总计 44 项。

## 原始证据

- [GitHub Actions 第三轮运行 35870733882](https://github.com/wumeizijiang519-beep/gesture/actions/runs/35870733882)。
- Job `107213819381`，Ubuntu 22.04，开始 `2026-09-23 13:57:47 UTC`，结束 `2026-09-23 14:00:17 UTC`，全部步骤成功。
- [源码附件 10755755058](https://github.com/wumeizijiang519-beep/gesture/actions/runs/35870733882/artifacts/10755755058)。
- [验收附件 10755415437](https://github.com/wumeizijiang519-beep/gesture/actions/runs/35870733882/artifacts/10755415437)。

下载后实际检查了 ZIP CRC、SHA256、源码包提交标识、两份 JUnit XML、五条 episode 的文件清单与日志，以及三项任务的 summary 和 metrics。

| 检查 | 观察结果 |
|---|---|
| 第一份 JUnit：ci-junit.xml | tests=44；failures=0；errors=0；skipped=0；19.423 s |
| 完整验收中的第二份 JUnit：acceptance/junit.xml | tests=44；failures=0；errors=0；skipped=0；18.880 s |
| 依赖与静态检查 | 固定依赖安装、pip check、compileall、ruff 全部通过 |
| 模型与真实静态图片 | 模型校验/CPU 推理通过；两张单手图片的检测及 PnP 通过，多手图片被拒绝 |
| CFR RGB 视频 | 30 行记录、30 行 ACTIVE；仅验证静态图片视频的通路 |
| 图形界面 | Xvfb 窗口冒烟检查通过，不是实体摄像头测试 |
| 三任务与数据管线 | 合成输入 → PnP → 控制 → IK → 物理仿真 → 记录 → 动作回放 → 校验 → NPZ → 指标图，通过 |
| 独立目录源码包 | 版本标识吻合；解压后 9 s 到达任务成功，270 行记录 |
| 本地独立复核 | 39 项不依赖 PyBullet/MediaPipe 的测试通过；不替代上面的完整 Ubuntu 测试 |

三任务统一 seed=7、30 Hz 控制、240 Hz 物理步、30 s 模拟时间，各 900 条记录：

| 任务 | success | 首次成功 active 时间 | IK 拒绝 |
|---|---|---:|---:|
| reach | true | 4.033333 s | 0 |
| path | true | 19.5 s | 0 |
| pick-place | true | 25.4 s | 0 |

抓放任务 cube_lifted=true；程序使用接触/摩擦，不绑定或瞬移物体。路径跟踪参考轨迹的 RMSE 为 0.0047049923346461015 m。这是合成闭环的结果，不是手部检测精度。

环境记录：CPython 3.10.21；NumPy 1.26.4；SciPy 1.13.1；OpenCV-contrib 4.11.0.86；PyBullet 3.2.7；MediaPipe 0.10.21。完整依赖见 requirements.lock 和验收附件中的 acceptance/environment.txt。

## 文件校验

```text
源码 ZIP 本身 gesture-source.zip：
fc287666f788192708a8774936e8c5a27e4097ea1eddd20ac0e4ea1e40f036a9

GitHub 源码附件外层 ZIP（包含源码 ZIP 和校验文件）：
5adfdf32cbb42cd8b16c5c554ce492483288e68b2b285f632bc14f22150a3976

GitHub 验收附件 ZIP：
83d90e0809868390d7c0f4ca25e58289201a6efb112e84e88c46008b937126a4
```

源码包含 SOURCE_REVISION.txt，值为上述已测试提交。源码包不含第三方依赖安装包、模型权重或这个在验收后补充的报告；首次安装仍需要联网。

## 未完成的现场实验

用户笔记本的摄像头权限/驱动、真实标定、掌宽测量、动态遮挡/照明鲁棒性、真实三维误差、实际端到端延迟、真人三任务成功率和 Windows 11 全流程尚未现场测试。不得把三条合成测试成功报告为真人遥操作 100% 成功率。按 README 和 docs/EXPERIMENTS.md 完成小组现场实验并保存失败记录。
