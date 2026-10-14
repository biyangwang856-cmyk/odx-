# 道路几何与硬接缝实验（2026-10-13）

本轮在 `1cbe2a7` 后继续，目标为高速道路半车、残影、标线波浪和西段补丁接缝。原生 2.5D、当前 OGS2 和全部历史试验均保留；新实验只写独立目录。代码测试通过不等于图像验收通过。

## 已核实的机制

- 原 OGS2 的每个 tile 独立选图并覆盖有效 RGB；halo 没有强制相邻 tile 采用同一来源，容易出现背景块缝。
- 原高度场在单张源照片的投影中仍使标线弯曲；已保存的同照片/同相机表面对照显示，平面投影明显减轻波浪。仍须验证道路高度和边界，不能把视觉直线当测绘真值。
- 当前数据的 `undistorted/tracks.csv` 保留了原 Brown 图像的测量坐标。独立验证时应从原轨迹经 `raw_camera.pixel_bearing` 再经去畸变相机 `project` 转换，不能直接将旧坐标当去畸变观测。本轮没有改写原轨迹或原重建文件。
- Orthority `FrameCameras.get(name)` 可复用并更新相机对象；不能在多个回调中保留这个可变对象。调用时按照片名重新取姿态，或使用独立不可变的相机矩阵。

## 新模块

`app/model_engine/ogs2_road_surface.py` 从独立照片三角化点拟合局部道路平面。固定世界网格划分拟合/留出点，再做拟合内 RANSAC；完整留出点都参与高度与原照片投影检查。只允许在训练支持域内输出高度。自动道路候选不会被标成已经人工确认的道路真值。

`app/model_engine/ogs2_road_seam.py` 在两幅同网格真实投影间寻找硬接缝。接缝的每个实际跨来源边必须位于双方有效、静态和几何已验证的区域，并避开双方车辆与阴影；标线未经连续性确认不能切过。输出只复制一个真实来源，不做车辆混合、隐式填补或自由形变。

## 研究依据与直接接入判断

- [Orthority](https://github.com/leftfield-geospatial/orthority) 提供现有相机/DEM正射接口。[issue 35](https://github.com/leftfield-geospatial/orthority/issues/35) 报告单图与 ODM 正射的静态点错位，未提供通用修复，不能把更换投影库当作几何修复。
- [LPAM，ICCV 2025](https://github.com/tlliao/LPAM_seam-cutting)：借鉴先检测坏缝、再局部处理的顺序。原 MATLAB/SIFT-flow 会改变局部几何，本轮没有直接引入自由形变。[论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Liao_Leveraging_Local_Patch_Alignment_to_Seam-cutting_for_Large_Parallax_Image_ICCV_2025_paper.pdf)。论文中的旧仓库用户名地址不可用，实际可用地址为 tlliao。
- [OBJ-GSP，AAAI 2025](https://github.com/RussRobin/OBJ-GSP)：对象/线结构约束值得借鉴，但官方 issue 2 确认当前一次拼两图，issue 6 说明默认全局缩放且未实现固定参考图，不能把 panorama 输出直接当作当前地理网格的正射。
- [OpenCV 图割接缝实现](https://github.com/opencv/opencv/blob/4.x/modules/stitching/src/seam_finders.cpp) 与现有 SciPy 最大流可复用；颜色相似本身不证明静态道路，仍需来源与对象约束。

## 验收分工

道路内部几何使用原照片独立静态留出点验证；与原图拼接只在最终接缝及邻域检查。旧正射内部可能本来就错误，不能强迫正确修复继续贴合它，但也不能取消最终边界检查。

本轮运行目录为 `/volume/wzhang/mesh-research-workspace/runs/mangrove-077-20261013/road-surface-20261013`。其中 `track-calibration-probe.json` 记录坐标转换问题，`multiview-tracks-01` 保存新增真实照片特征及三角化数据，`multiview-fit-01` 保存完整留出结果和单源候选。区域搜索带是本航次试验配置，并非已经完成通用自动道路识别。

## 回退

实验前标签：`checkpoint/before-road-surface-20261013`（`1cbe2a7`）。更早整轮车辆实验前标签：`checkpoint/before-corridor-source-20261013`（`604aabe`）。只切换代码不会删除基线影像和新实验结果。

```bash
cd /volume/wzhang/mesh-research-workspace/mesh-research
git switch --detach checkpoint/before-road-surface-20261013
# 返回当前开发
git switch feature/by-dev
```

没有推送 GitLab，没有合并主分支。
