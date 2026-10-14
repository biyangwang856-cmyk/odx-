# 新航拍数据车辆畸变、重复和残影：接入预研

核查日期：2026-10-13。该预研在出图前完成，现已补充首轮实测状态：405张新DJI M4E照片的原生2.5D及恢复OGS2正射均已完成。路口和北侧道路有局部外观收益，南侧高速仍有车体破碎、灰色残留及车道线畸变。下面的方法排序是接口、成本和风险下的工程判断，不是实测排行榜。道路平面与整片选图试验没有获得足以安全写入全幅的独立证据，详见 `road-trial-decision-20261013.md`；两份正式TIFF保持不变。

## 先区分原因

- 同一辆车在不同时间的位置被一起拼进正射：需要时间一致性与源图选择；增加点云密度不能单独解决。
- 一辆车被多个源图切开、羽化，或旧位置及其阴影留在对象掩膜之外：需要完整冲突区域单源替换，切缝绕开车身、阴影及未知变化。
- 静止车辆和道路标线同时错位，或单源重投影也严重拉伸：首先是相机 / 表面 / 视角映射问题，不能用消重掩盖。
- 原照片车身本来运动模糊或被遮挡：选另一张清晰真实照片；没有完整观察时保留原输出并标记不支持，不能生成式补车。
- 多辆同色同型号车不构成重复证据。仅看成图的车数下降不构成改善证据。真实投影阴影也不等于残影。

## 方案排序与证据

| 排序 | 方案及公开依据 | 当前接入点 | 预计收益和适用条件 | 主要成本 / 风险 |
|---|---|---|---|---|
| 1 | 道路冲突区域统一真实照片时刻，车身与阴影禁止切缝，静态背景允许有限过渡。借鉴 [OpenCV 图割切缝源码](https://github.com/opencv/opencv/blob/4.x/modules/stitching/src/seam_finders.cpp) 与 [MVS-Texturing，ECCV 2014](https://github.com/nmoehrle/mvs-texturing) 的离散源图选择思路。 | 已有 `ogs2_road_coherent.py`、`ogs2_joint_policy.py`、`ogs2_joint_geometry.py`、`ogs2_joint_seam.py`、`ogs2_apply_patch.py`；直接使用已校准源图、2.5D 表面和当前正射网格。 | 若静态配准正确，最直接消除同一区域混入不同时间的车身 / 阴影和切碎现象。先做一个真实道路 ROI 对照，不要求先破解全局身份。效果为预测，需新场景验证。 | 一张照片可能覆盖不了整个冲突范围；道路坡度、立交、多层场景不能都当平面；道路块边界仍可能重复，需扩展上下文并检查。已有局部成功不能外推为全场解决。 |
| 2 | 在地面坐标与真实曝光时间上做保守跨照片关联，再做轨迹整体单源保留。借鉴 [ByteTrack，ECCV 2022](https://github.com/FoundationVision/ByteTrack)、[TrackTrack，CVPR 2025](https://github.com/kamkyu94/TrackTrack)、[BoT-SORT](https://github.com/NirAharon/BoT-SORT) 的分级检测关联 / 多线索关联。 | 扩展 `ogs2_temporal.py` 和新场景 `observations.json`；由 SfM + DSM 投影消除无人机自运动，再输入时间差、世界位置、尺寸、行驶方向、外观；接 `joint / road_coherent` 整块输出。 | 对跨拼接块边界、同一条道路上的重复位置有潜力，比逐对象独立选图更完整。检测与分割模型已在服务器，可先只借鉴关联算法。 | 官方追踪器是连续视频，主要验证于行人 MOT；不能原样用帧 IoU / 30 fps 处理跨航带稀疏照片。长时间间隔、相同车、遮挡和车辆高度投影偏差可造成错误身份。外观只做辅助，歧义则不删除。 |
| 3 | 增强车身—阴影联合保护，解决车体被替换后旧阴影 / 残影留在外围。论文依据 [ViShadow，IEEE TIP 2024](https://github.com/HarryHsing/Video-Instance-Shadow-Detection)；更贴 SAM 接口的候选 [OPSeg，IJCNN 2025](https://github.com/deyang2000/OpSeg)，后者不是顶会论文。 | 只把推断阴影作为候选掩膜，与原照片间实际变化支持求联合 / 交叉核实；进入 `observed_dynamic_support()`、`choose_whole_source()`，不直接改 RGB。 | 如果残影是掩膜遗漏的投影阴影，这条比换 SfM / MVS 更对症。还必须取原图真实阴影和车体一起输出；不得把所有阴影去掉。 | ViShadow 依赖旧 PyTorch / Detectron 系环境；稀疏照片、俯视小车与 SOBA-VID 域不同。OPSeg 仓库当前只有 README 和权重下载入口，不能称完整稳定推理工程；权重需另存服务器，需单独 SAM 2.1 环境 / 配置。没有车辆畸变或身份消重保证。 |
| 4 | 静态几何验证增强；有道路纹理弱、SIFT 证据不足时借 [RoMa，CVPR 2024](https://github.com/Parskatt/RoMa) 做独立对应，检查源图正射是否错位。 | 可作为 `independent_static_evidence()` 的可选匹配提供者，必须屏蔽车辆、阴影和动态区域，保留独立空间留出验证。先用于诊断，不私自非刚性扭曲地理坐标。 | 对确实由几何误差引起的拉伸有助于发现和定位问题，但不是动态车重复的独立解法。静态错位已确认时，应回到相机 / DSM 解决。 | 新权重及 DINOv2 依赖，当前版本与论文原版存在分辨率和采样差异；不能把特征对应强行当真实几何。更密的匹配不自动意味着更准。 |
| 5 | 通用拼图论文 [Seam Estimation Based on Dense Matching，CVIU 2025 卷](https://github.com/zhihao0512/dense-matching-image-stitching) 或 [DSFN，NeurIPS 2025](https://github.com/DLUT-YRH/DSFN)。 | 可以借缝代价思路；不宜直接替换正射生成器。 | 针对视觉拼图中的视差、切缝有研究价值。未证实能保留我们标定后的米制坐标并处理动态车辆身份。 | 前者官方只提供成对示例，Windows DLL，需要重新构建；多图问题仍未回答。后者需要 DepthAnything + Warp + Fusion 权重，视觉 warp 会改变正射几何。两者根目录未见明确许可证，不能假定许可完备。 |

## 维护、许可与离线依赖

2026-10-13 的 GitHub API 查询全部返回公开 API 限流 403，父任务记录于本轮作业 `research-repo-status.json`；不绕过、不索要 token。以下是公开文件 / issues 页面事实，不能用星数或“还有 issue”冒充维护活跃。

| 项目 | 可核实状态 | 许可 | 最小离线依赖 |
|---|---|---|---|
| OpenCV | 官方 [release 页面](https://github.com/opencv/opencv/releases) 持续发布；项目已有 OpenCV 4.10，不需要为本方案升级。 | 官方 [Apache 2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE)。 | 已安装 NumPy、SciPy、OpenCV；无需额外学习权重。 |
| MVS-Texturing | 公开源码与构建说明可访问；具体最后维护日期未核实。ODX 已有本类纹理阶段，不应重新下载整个工具链当作新增算法。 | README 明确 BSD 3-Clause。 | 只借选图思路可不引入新增运行依赖；复制源码须保留许可。 |
| ByteTrack / TrackTrack | 官方源码、论文归属可验证；当前最后提交日期未核实。TrackTrack 为 2025 CVPR 正式作者实现，环境说明 Python 3.8 / Torch 1.13。 | 均 MIT，见各根目录 LICENSE。 | 若只适配关联核可复用现有检测观测，不必下载 YOLOX / 行人 ReID；全套官方流程则需要对应训练模型与旧环境。 |
| BoT-SORT | 安装兼容和 GIS 关联 issues 仍开放，持续维护未证实；不建议为已具备检测结果的项目安装全套 2022 环境。 | [MIT](https://github.com/NirAharon/BoT-SORT/blob/main/LICENSE)。 | 关联核、真实运动补偿；如加 ReID，需要车域权重及域验证。 |
| SAM 2 | 官方 [README](https://github.com/facebookresearch/sam2) 有 SAM 2.1 与 2024-12-11 视频预测器更新；最后维护日期未核实。 | 模型与主体代码 Apache 2.0；可选连接组件第三方许可另列。 | 本轮旧版本 image predictor 已可用。新 video predictor 特性需固定新源码 / checkpoint，不能静默替换基线。 |
| ViShadow | 官方 demo 和 GitHub release 权重可访问，README 仍列 training / evaluation TODO；最后维护日期未核实。 | 根目录未看到明确 LICENSE，不能默认商用许可。 | Python 3.8、Torch 1.13 / CUDA 11.7、私有安装 `visd`、`model_final.pth`，需要适配新 H200 运行环境。 |
| OPSeg | 当前 GitHub 根目录只有 README；模型可由 ModelScope / Google Drive 下载，0 issue 不证明没有问题；完整推理入口和维护未核实。 | README 声明 Apache 2.0，但 LICENSE 链接和模型卡需下载前核对。 | SAM 2.1 base+ 与微调权重；可以无生成式 RGB 编辑，仅提供保护掩膜。 |
| RoMa | 官方当前代码包含 Python 3.12 流程、Tiny RoMa 与复现差异说明；具体最后提交日期未核实。 | 主体 MIT，DINOv2 Apache 2.0。 | 新匹配权重及其基础特征权重；GPU 只需本地推理，下载走 CPU 共享盘。 |
| dense-matching / DSFN | 官方代码可访问；根目录未见 LICENSE，最后维护日期未核实。 | 未核实，不等于可以任意复制。 | LoFTR / FlowFormer 或 DepthAnything / Warp / Fusion 及其权重；本项目首轮不需要。 |

## 官方 issues 给出的限制和解决证据

1. **ByteTrack 低帧率**：[issue #64](https://github.com/FoundationVision/ByteTrack/issues/64)。作者明确回应纯 Kalman 在低帧率不好，建议结合 ReID。该问题关闭不等于发布了适用于本任务的解决版本。这里的照片间隔和跨航带回访比连续视频更严苛，必须用曝光 `Δt` 与地面投影关联；换帧率参数还不够。官方 [demo](https://github.com/FoundationVision/ByteTrack/blob/main/tools/demo_track.py) 还默认排除宽高比大于 1.6 的目标，这是行人过滤，对横向公交 / 货车不适合，借核心时不能保留。
2. **BoT-SORT GIS 与相机运动**：[issue #122](https://github.com/NirAharon/BoT-SORT/issues/122) 提出带位置 / 姿态的 Kalman 修正，页面没有官方完整解法。建议直接用现有 SfM / DSM 校准投影，而非低纹理路面的 ORB 估计当作真值。
3. **BoT-SORT 空描述子 / knn 不足**：[issue #124](https://github.com/NirAharon/BoT-SORT/issues/124) 记录运行崩溃并提出用户防护；仍 open，不能称作者已验证修复。对应工程处理：`descriptor is None`、不足两匹配、无共识时拒绝关联，不能默认为零运动。
4. **BoT-SORT 旧依赖**：[issue #131](https://github.com/NirAharon/BoT-SORT/issues/131) 与本轮 Ubuntu 24.04 / Python 3.12 完全同类：`onnxruntime==1.8.0` 无 wheel。回复建议旧 Python，另有人拟升级依赖，但不是合并修复证据。我们可借关联模块，不必引入这一套 detector / ONNX 环境。
5. **SAM 2 新对象与内存状态**：官方 README 指出 2024-12-11 才更新为独立逐对象推理，允许追踪开始后加新对象；旧版不能假定支持同特性。若为视频路径升级，CPU offload 的 [issue #489](https://github.com/facebookresearch/sam2/issues/489) 由 [PR #490](https://github.com/facebookresearch/sam2/pull/490) 关闭，固定包含修复的版本并验证 CPU / GPU tensor device。当前单张 image predictor 不需要为这个问题升级。
6. **通用拼图多图支持**：[dense-matching issue #1](https://github.com/zhihao0512/dense-matching-image-stitching/issues/1) 直接问能否超过一对，仍 open 且未见答复。不要把 pair demo 说成 405 张地理正射完整流程。
7. **RoMa 尺度与随机性**：官方 [复现说明](https://github.com/Parskatt/RoMa#reproducibility) 明确当前 `scale_factor`、PyTorch、随机采样 / RANSAC 和 GP 求逆实现较论文有变。若启用要固定源码、输入尺寸、权重、随机种子，进行留出几何检查；不能以更密匹配替代精度证明。

## 最优执行路线：先方案 1，小范围公平对照，再有证据地增加方案 2 / 3

1. 保存当前 2.5D + OGS2 代码 tag、输出 TIFF SHA、运行参数、源图和相机 / 表面 SHA；原 TIFF 只读，候选另存，不 push / merge。
2. 看全局图与多个 100% 分辨率局部，从成图与源照片确认“真实存在多车、重复位置、车身破碎、真实阴影、旧阴影残留、源图模糊、静态配准错位”各是哪类。选一个拥挤道路、一个公交 / 货车、一个停车区作为固定验证 ROI；不要从旧 FT-070 拷贝坐标或身份。
3. 原相机与 2.5D DSM 投影各候选照片到**同一正射像素网格**，所有有效性 / 遮挡 / 映射畸变检查保存。用远离车与影子的静态标线、路缘、护栏做独立几何证据；未通过则候选不写入，不靠像素非刚性 warp 隐藏错位。
4. 合并 ROI 中旧 / 新所有车身、近邻实际变化、车影和边界未知区域。按道路连通冲突范围选择能完整覆盖的一张真实照片，保持整辆车与其原有真实阴影；源图没有车的实际路面可以用于消除旧位置，但不能以“少几辆车”为目标抹掉静止车 / 未知车。
5. 硬保护区域采用原源 RGB 重投影，一律不羽化、不补画；静态、已验证的外圈才允许颜色过渡。未知区域越过有效覆盖或对象跨边界，扩大范围、联合相邻块或拒绝。记录每个输出像素的源图 index、权重、曝光时间和接受 / 拒绝原因。
6. 如果单张源图确实无法覆盖整个动态轨迹，再启用方案 2 的**局部短时关联**：用时间差、投影位置不确定性、同一时刻排他性、车类 / 尺寸 / 行驶方向；仅关联唯一可信匹配，长间隔 / 同色相邻 / 遮挡不强制续 ID。停车区默认保护真实静止车，不因为分割和车数变化判它是动目标。外观模型只作为辅助，不作为删车唯一依据。
7. 若已经取整段同源，车影仍遗漏，才测试方案 3。检测器仅扩充保护掩膜；预测阴影须受真实观测变化与邻近关系约束，真实车辆阴影随车保留。不能启用学习型去阴影 / diffusion 填路面的 RGB 生成功能。
8. 在同一 ROI、网格、3 cm 像素、同一缩放 / 曝光展示下比较基线、OGS2、道路候选及所选真实源图。报告逐例重复 / 破碎 / 残影变化，车辆保留与跨边界检查，静态对应误差、alpha 有效范围、源图硬区域一致性、接缝变化。无法从多张原图建立身份的地方标“无法确认”，不要给伪精确全局重复率。
9. 只把通过实际车辆与阴影检查、静态几何和覆盖检查的候选推广。每个阶段独立 commit / tag；无明显收益或损失停车车辆，保留实验结果和原因，恢复上一 tag / 主用输出路径。

这里没有找到“直接读本项目 405 张照片、保证车辆全局去重并输出测量可信正射”的完整现成顶会开源工具。推荐的是已有校准正射框架中的具体源图约束与必要组件接入，不把论文视频指标当作本场景效果保证。
