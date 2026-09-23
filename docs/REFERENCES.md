# 参考工作、借鉴范围与第三方资产

资料核对日期：2026-09-23。论文与仓库可能继续更新，应在课程报告中记录访问日期。以下是上游公开来源，不代表本项目实际调用了上游全部模块。

## EgoInfinity

EgoInfinity: A Web-Scale 4D Hand-Object Interaction Data Engine for Any-View Robot Retargeting and Video-to-Action Robot Learning.

- 论文：https://arxiv.org/abs/2606.17385
- 官方仓库：https://github.com/Rice-RobotPI-Lab/EgoInfinity
- 官方方法说明：https://github.com/Rice-RobotPI-Lab/EgoInfinity/blob/main/docs/PIPELINE.md

借鉴模块化感知、几何处理、质量检查、重定向与数据管理。不同点：本项目用固定相机、掌宽先验和轻量手部模型，且只操作仿真几何物体；不包含单目稠密深度、物体分割、三维物体重建、接触恢复、大规模数据引擎或其训练策略。没有下载或分发 EgoInfinity/MANO 资产。

## MINT

MINT: A Unified Model for World-Space Camera and Hand Motion Estimation from Scalable Egocentric Pipeline Supervision.

- 官方仓库：https://github.com/wuji-technology/wuji-ego-mint
- 论文索引：https://arxiv.org/abs/2609.04958

借鉴显式区别相机空间/手部空间/世界空间，以及通过几何变换形成一致表示。不同点：本项目不运行 MINT 的统一网络、不预测移动相机外参、不做 SLAM，也不从第一视角视频恢复长时世界坐标轨迹。不能把本项目的 fixed-camera PnP 称为 MINT 复现。

## 实际运行的库与方法

- MediaPipe Hand Landmarker：https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- Python Tasks API：https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- OpenCV PnP：https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
- PyBullet 官方仓库：https://github.com/bulletphysics/bullet3
- One Euro Filter 作者页面：https://gery.casiez.net/1euro/

MediaPipe 模型是版本化 hand_landmarker/float16/1；下载地址与 SHA256 写入 vision.py。代码会检查哈希，不从可变 latest 地址下载，不在仓库复制模型文件。MediaPipe、OpenCV、PyBullet/pybullet_data、NumPy、SciPy 等第三方代码、URDF/网格和权重保留各自许可证；本仓库 MIT 只适用于原创代码，不能用其覆盖第三方资产条款。重新分发完整离线包前检查上游模型与资源许可。

使用真实视频应获得拍摄者/参与者授权。原始视频和标定文件默认被 .gitignore 排除；不要上传同学面部、桌面屏幕或其他无授权私人素材。合成演示是程序生成的几何测试，不是人类数据。
