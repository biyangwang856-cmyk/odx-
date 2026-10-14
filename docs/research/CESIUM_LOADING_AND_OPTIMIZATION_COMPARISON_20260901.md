# Cesium 读取与优化对比：Surveying Frontend vs Mesh Research

## 1. 结论先行

两边的差异不是简单的“某几个 Cesium 参数不同”，并且 `surveying-frontend` 内部本身存在两条
3D Tiles 加载路径：

1. `surveying-frontend` 的普通 `/scene-3d` 默认模型路径偏向通用浏览体验；
2. `surveying-frontend` 的 Scene Runtime mesh 图层路径偏向高质量业务查看；
3. Mesh Research Viewer 在同一页面中比较 Terra 与 Remesh。

公平比较的控制变量不是某个参数必须取固定大小，而是左右模型在同一次运行中必须使用完全相同的
Cesium 配置。SSE、cache、dynamic/foveated、preload、culling 和请求调度都可以继续优化；只要两边
共用同一个 options 对象，并在报告中记录实际值，绝对数值本身不会破坏控制变量。

最直接影响观感的参数分别是：

| 路径 | 默认/固定 SSE | dynamic SSE | foveated SSE | 显式 cache budget | 渲染策略 |
| --- | ---: | --- | --- | --- | --- |
| Frontend `/scene-3d` 通用路径 | 24 | 开 | 开 | 未指定 | `requestRenderMode=true` |
| Frontend Scene Runtime mesh | 1 | 关 | 关 | 1 GiB cache + 1 GiB overflow | `requestRenderMode=true` |
| Mesh Research（优化后共享配置） | 8，可由 URL 调到 1-128 | 开 | 开 | 每模型 512 MiB + 512 MiB overflow | 连续渲染和持续采样 |

原始 Mesh Research 默认 SSE 32 且没有显式 cache/foveated/preload 配置，这足以解释它与旧 Viewer
在质量和速度上的体感差异。优化后的新 Viewer 不复制旧版本的数值约束，而是把一组新的共享配置
同时应用到 Terra 与 Remesh：默认 SSE 8、每模型 512 MiB cache + overflow、dynamic/foveated 和
flight preload 开启，`preferLeaves` 关闭，并统一关闭 children-bounds 与移动中请求裁剪。

## 2. 对比范围与证据

- Surveying Frontend：`surveying/surveying-frontend`，观察 revision
  `9f23209ce68d40baee38a00607eac26814224b68`；相关文件在任务开始时无未提交修改。
- Mesh Research：`surveying/ai-models/mesh-research`，观察 revision
  `89441898b0e7e30f3162aed41c158e59b5123739`。
- 证据优先级：实际代码 > 测试/可执行探针 > 文档。
- `confirmed` 表示代码直接证明；`inferred` 表示由参数语义或调用关系推导；`uncertain` 表示需要
  浏览器或生产路由观测才能确认。

## 3. 三条真实加载链路

### 3.1 Frontend 普通 `/scene-3d` 路径

```text
CesiumScenePage
  -> 查询参数 ?tileset=... 或 VITE_CESIUM_DEFAULT_TILESET_URL
  -> CesiumScene useEffect
  -> loadTileset(viewer, url)
  -> Cesium3DTileset.fromUrl
```

证据：

- 地址优先级在
  `surveying-frontend/services/portal/src/pages/CesiumScenePage.tsx:67` 和 `:87`。
- React 生命周期加载、disposed 保护和卸载删除在
  `surveying-frontend/services/portal/src/components/cesium/CesiumScene.tsx:289`。
- 该路径直接使用 `loadTileset` 默认值，没有传 mesh 专用覆盖项。
- 通用默认值在
  `surveying-frontend/services/portal/src/services/cesium/tilesets.ts:27`：SSE 24、dynamic/foveated
  开、渐进分辨率 0.3、`preloadFlightDestinations=true`、`preferLeaves=true`。

这条路径强调“快速出现可用画面”和飞行时渐进加载，不是最高几何质量路径。`confirmed`。

### 3.2 Frontend Scene Runtime mesh 路径

```text
SceneRuntimeLayout checked layer tree
  -> mountSceneLayerOverlay
  -> mountMeshTilesetLayer
  -> acquireMeshTileset(resource.id)
  -> 先鉴权 fetch + JSON 校验
  -> createAuthorizedResource
  -> loadTileset(viewer, Resource, mesh profile)
```

证据：

- Scene Runtime 根据图层树增量 mount/unmount：
  `surveying-frontend/services/portal/src/components/scene-runtime/SceneRuntimeLayout.tsx:248`。
- mesh 分派进入 `mountMeshTilesetLayer`：
  `surveying-frontend/services/portal/src/components/scene-runtime/sceneLayerOverlays.ts:100` 和 `:219`。
- resource ID 引用计数 cache 与 120 秒延迟淘汰：
  `surveying-frontend/services/portal/src/services/cesium/meshTileset.ts:25`、`:43`、`:284`、`:406`。
- 加载前先携带组织作用域/同源 cookie 拉取 manifest 并验证 JSON：
  `surveying-frontend/services/portal/src/services/cesium/authorizedResource.ts:41` 和
  `meshTileset.ts:307`。
- mesh profile 在 `meshTileset.ts:326`：SSE 1、1 GiB cache、1 GiB overflow、隐藏时预取，
  dynamic/foveated/渐进分辨率/children bounds culling/移动中请求裁剪全部关闭。

这条路径是质量优先、内存换细节的业务 profile。`confirmed`。

### 3.3 Mesh Research 对比路径

```text
FastAPI startup validation
  -> /api/config 返回左右模型同源 URL
  -> Viewer 根据 mode 计算 requiredModels
  -> Cesium3DTileset.fromUrl
  -> registerTileset 记录 load/visible/failure 状态
  -> split / flicker / 单模型显示
```

证据：

- FastAPI 启动前验证左右 `tileset.json` 与 Viewer build：
  `services/viewer_api/config.py:64`。
- `/terra`、`/a1` 由同一 FastAPI 静态服务发布，配置接口返回同源地址：
  `services/viewer_api/app.py:43` 和 `:71`。
- 默认只加载当前 mode 需要的模型，进入 split/flicker 时按需加载缺失模型；并发调用复用同一
  Promise，失败后允许重试：`services/viewer/src/main.ts:294`、`:327`、`:357`。
- 两个模型共用同一个 `commonOptions`；优化后默认 SSE 8（URL 可调 1-128），cache、dynamic、
  foveated、flight preload、culling 和请求策略均由这个对象统一传入：
  `services/viewer/src/main.ts:305` 和 `:320`。

这条路径强调对比条件一致和运行时可观测，而不是复用业务门户的缓存、鉴权和图层生命周期。
`confirmed`。

## 4. 关键差异矩阵

| 维度 | Surveying Frontend | Mesh Research | 直接影响 | 置信度 |
| --- | --- | --- | --- | --- |
| 数据来源 | 第三方系统/API、图层资源记录、组织作用域 | 本地/JuiceFS publication，由 FastAPI 同源挂载 | Frontend 必须处理鉴权和动态资源；Research 假设数据已发布 | confirmed |
| Cesium 版本 | `^1.140.0`，允许兼容范围内升级 | `1.144.0`，精确锁定 | 私有字段、默认 cache 和 loader 行为可能有版本差异 | confirmed |
| manifest 预检 | mesh 路径先 fetch、检查 HTTP/空响应/JSON | 服务启动时检查文件存在，浏览器直接 `fromUrl` | 错误暴露位置不同 | confirmed |
| 加载粒度 | 任意业务图层，按资源 ID mount/unmount | 固定左右两套 publication | Frontend 生命周期更复杂；Research 对比更稳定 | confirmed |
| tileset cache | resource ID + viewer + refCount，120 秒延迟淘汰 | 页面内 Promise 去重；无跨卸载 cache/淘汰 | 重开图层时 Frontend 更容易复用 | confirmed |
| SSE | 通用 24；mesh 1 | 默认 8，可调 1-128，左右同步 | 任意取值都可公平比较；更小通常请求更多细节 | confirmed |
| dynamic/foveated | 通用路径开；mesh 路径关 | 左右统一开启，`preferLeaves=false` | 两边接受相同的动态与视中心优先规则，同时保留 HLOD 粗层快速稳定能力 | confirmed |
| cache 容量 | mesh 显式 1 GiB + 1 GiB overflow | 每模型 512 MiB + 512 MiB overflow | 左右预算相同；双模型合计峰值仍需实测 | confirmed/inferred |
| hidden preload | mesh 开 | 关 | Frontend 图层切换更快但后台耗资源；Research 单模型模式更省 | confirmed |
| request 调度 | mesh 关闭移动裁剪；未见全局 per-server 覆盖 | 不覆盖 Cesium 全局上限；左右都关闭移动裁剪与 leaf 偏好 | 两模型共享相同调度条件 | confirmed |
| 渲染循环 | `requestRenderMode=true`，显式 `requestRender()` | 连续 RAF、postRender、250ms/1s 指标采样 | Frontend 空闲功耗低；Research 更适合测 FPS 和可见瓦片 | confirmed |
| 地球/底图 | Globe + 天地图 +雾/太阳/月亮 | Globe 隐藏、纯色背景、无底图 | 深度、背景、相机定位体验不同 | confirmed |
| 相机定位 | 经纬度/EPSG/tileset volume 推断，两段 warmup 后精确落点 | 基于已加载模型 bounding sphere 的固定相对 preset | Frontend 更数据集无关；Research preset 更适合固定实验视角 | confirmed |
| ready 判定 | 通用页 `fromUrl` 后即 ready；Scene Runtime 可等待 globe + mesh idle | `initialTilesLoaded`、pending/processing、20 秒 UI fallback | “页面已出现”和“瓦片稳定”的定义不同 | confirmed |
| 失败处理 | React disposed、资源错误状态、cache cleanup | load Promise 可重试、tileFailed 计数、顶层错误 overlay | 两边覆盖的是不同生命周期 | confirmed |
| 诊断 | camera HUD、比例尺、业务状态；缺少 mesh 直测 | FPS、队列、可见/已载瓦片、三角形、内存、网络、浏览器回传 | Research 更适合性能实验 | confirmed |
| 双模型比较 | 没有同相机 split/flicker 的专用契约 | split/flicker/单模型、测量与成对截图 | Research 更适合 A/B 质量验证 | confirmed |
| Cesium 构建 | `vite-plugin-cesium` + Cesium manual chunk | 手动复制 Assets/Workers，未拆 Cesium chunk | Research 首屏 JS 更大 | confirmed |

## 5. 为什么视觉和加载体感会不同

### 5.1 最大因素：原始 SSE 1 对 SSE 32

SSE 越小，Cesium 越早细分到子瓦片。Frontend Scene Runtime mesh 固定为 1，而原始 Mesh Research
默认为 32，相差 32 倍阈值。新 Viewer 改为默认 8，但公平性的关键仍然只是左右模型在同一运行中
使用相同 SSE、相机、viewport 和 tileset options。`confirmed`。

建议复现实验：在 Mesh Research 使用 `?mode=split&sse=1`、`4`、`8`、`16`、`32`，每一档都让
Terra 与 Remesh 共用该值，再记录稳定时间、请求数、峰值内存和可见三角形。

### 5.2 Frontend mesh 用内存换细节

Frontend mesh 显式给出 1 GiB cache 和 1 GiB overflow，并关闭多种裁剪；Research 没有显式预算。
因此相同相机往返时，Frontend 更可能保留高细节内容，但首次下载、GPU/CPU 内存和移动中请求也会
更高。这里“最多约 2 GiB 可用预算”是基于 Cesium cache/overflow 参数含义的推断，实际峰值必须
用浏览器测量确认。`inferred`。

### 5.3 Frontend 会预热相机目的地

Frontend 的通用 loader 设置 `preloadFlightDestinations=true`，Scene Runtime 又对首次 focus 执行
500 m 档预热、等待瓦片 idle，再进入 100 m 最终视角；warm target 后才走直接飞行：
`SceneRuntimeLayout.tsx:47`、`:503`、`:681`、`:777`。Research 直接用 bounding sphere preset 飞行，
浏览器探针再等待 pending/processing 归零。因此两边“点击后多久清晰”的时间线不是同一协议。
`confirmed`。

### 5.4 Research 的连续诊断本身需要连续渲染

Research 每帧采 FPS，并在 postRender 收集 visible tile，再每 250 ms 刷新指标；同时每秒向 API
回传诊断：`services/viewer/src/main.ts:271`、`model-state.ts:131`、`diagnostics.ts:39`。如果直接照搬
Frontend 的 `requestRenderMode=true`，空闲时 postRender/RAF 行为和 FPS 含义都会改变，历史结果将
不可直接比较。`confirmed`。

### 5.5 构建拆包也影响首屏

Frontend 使用 `vite-plugin-cesium`，并把 Cesium 独立成 manual chunk：
`surveying-frontend/services/portal/vite.config.ts:16` 和 `:57`。Research 只在 build 结束复制 Cesium
runtime 目录：`services/viewer/vite.config.ts:9`。在 revision `8944189` 上执行 `npm run build` 的本地
观测中，Research 主 JS 为约 4,185 kB（gzip 约 1,123 kB），Vite 发出大于 500 kB warning。这个差异
影响应用 shell 首屏，不等同于 3D Tiles 内容加载速度。`confirmed`。

## 6. 各自已有的优化

### 6.1 Surveying Frontend 已有

- 按需渲染，降低静止场景 CPU/GPU 使用。
- 通用与 mesh 专用 profile 分离。
- 组织作用域 header、同源 cookie 与 Cesium `Resource` 鉴权。
- manifest 预检和可读错误。
- resource refCount、延迟淘汰和隐藏复用。
- region/sphere/box + root transform 的 focus target 解析。
- EPSG:4326/4547/3857/900913/32650 转换和 metadata fallback。
- 两段相机 warmup、globe/mesh idle 等待和 warm target cache。
- Cesium 独立 chunk 与插件化 runtime 资源复制。

### 6.2 Mesh Research 已有

- 左右模型共用相机、viewport、SSE 与加载参数。
- 单模型启动，进入比较模式时再懒加载另一模型；并发去重且失败可重试。
- split、flicker、单模型和固定视角 preset。
- URL 控制 SSE、preview 和 medium publication。
- preview/medium tileset 服务端派生，并验证 Range 请求和 MIME。
- pending/processing、初始加载耗时、tile failure、可见/已载 tile、三角形、内存和网络指标。
- 可重复 Playwright 路径、成对截图、稳定时间和资源流量报告。

## 7. 建议移植到 Mesh Research 的能力

### P0：保持唯一共享 options

Terra 与 Remesh 必须只从同一个 `commonOptions` 创建 tileset；UI 的统一 SSE 滑块也必须同时更新
两个已加载 tileset。可以自由调整数值寻找新版本的质量/速度平衡，但每份报告必须记录 Cesium
版本、完整 options、viewport、相机和输入 revision。禁止给左右模型增加分支配置。新 Viewer 统一
设置 `preferLeaves=false`，避免主动偏向叶节点；但受控实测表明，把双方 SSE 同时设为 8、16、32，
以及切换 `preferLeaves` 前后，Remesh 在 overview 中都请求约 271 个 GLB，30 秒内均未完全 settle。
因此当前主要加载量不能归因于 SSE 绝对值或 leaf 偏好，后续应检查 publication 的 geometricError、
层级覆盖和内容粒度。

### P1：移植数据集无关的 focus target 解析

Research 当前 preset 依赖一个模型 bounding sphere 和固定 east/north 偏移。可复用 Frontend 对
region/sphere/box/root transform 的解析，先得到 publication 的稳定中心，再从半径和视口推导相机
距离。EPSG metadata fallback 仅在 publication 仍携带投影坐标时启用。

收益：减少针对 LGLGLG 固定视角的隐含偏置，使新数据集无需手工调 preset。`recommended`。

### P1：引入有界 cache 参数并进入诊断报告

为 Research profile 显式设置 `cacheBytes` 和 `maximumCacheOverflowBytes`，并把值写入配置接口和
输出报告。不要直接采用 1 GiB + 1 GiB；先测浏览器内存上限、数据规模和两模型同时加载峰值。

收益：结果可复现，也能解释相机往返后的二次加载差异。`recommended`。

### P1：补齐前端构建拆包

至少把 Cesium 拆成独立 chunk，或评估 `vite-plugin-cesium` 后保持 Assets/Workers 路径兼容。用
构建产物大小和首次应用 shell 加载时间验收，不要把 bundle 变小误报为 tileset 优化。

### P2：统一稳定判定 helper

把 Playwright 中重复的 pending/processing quiet streak 与 Viewer 的 `initialTilesLoaded` 合并为一个
公开、可测试的 readiness contract：

```text
tileset 已创建
  -> 首批可见 tile > 0
  -> pending == 0 && processing == 0 连续 N 次
  -> 相机未移动
  -> 记录 settled_at
```

避免只依赖私有字段 `_depth`、`_header`、`_boundingVolume`；这些字段在 Cesium 升级时可能变化。

## 8. 不建议直接照搬的优化

- **左右模型使用不同 SSE/cache/裁剪值**：这才会真正破坏控制变量。
- **只记录 SSE、不记录其余 options**：dynamic/foveated/preload/culling 同样会改变请求和选择过程。
- **`requestRenderMode=true` 用于基准测试**：会改变 Research 当前 FPS/postRender 采样协议。
- **把 `preferLeaves=true` 当作已证明的速度优化**：当前 A/B 探针没有观察到请求量下降。
- **只给一边做 focus warmup**：预热会提前下载最终视角附近瓦片；若两边共同预热仍然公平，但必须
  把整轮标记为 warm run。
- **业务鉴权和 120 秒 resource cache**：仅在 Research Viewer 进入多用户长期服务时需要；当前本地
  publication 服务不应提前引入组织身份复杂度。

## 9. 测试与证据缺口

| 缺口 | 现状 | 建议 |
| --- | --- | --- |
| Frontend mesh 直接回归测试 | 未检索到 `meshTileset.ts`、`loadTileset` 或 Scene Runtime focus 的直接 test | 为 profile、cache refCount、manifest error 和 focus parser 增加 unit/contract tests |
| Research Viewer 单元测试 | API preview/range/diagnostics 有 2 个测试；Viewer TS 主要靠 type-check 和 Playwright 工具 | 为 profile 解析、lazy model loading 和 readiness contract 增加纯函数/浏览器测试 |
| 同一 publication 横向实测 | 本文只比较代码和已有构建/浏览器证据 | 用同一 tileset、相机、viewport 跑 SSE 1/4/8/16/24/32 矩阵 |
| Frontend 实际活跃路径 | 代码确认两条路径都存在，但本文未追踪生产用户当前进入哪条 | 在真实 route 记录 loader/profile 名称并采集一次网络/内存 trace |

## 10. 最小横向实验

为了把“感觉有区别”变成可复现结论，建议下一步只做一个受控矩阵：

1. 选择同一个 `tileset.json`，禁用底图和额外业务图层；
2. 固定 Cesium 版本、Chromium、1200x760 viewport 和五个 camera preset；
3. 分别使用 `comparison` profile 与 Frontend 通用/mesh 参数；
4. 每组做 cold run 和 warm run；
5. 记录首个可见 tile、稳定时间、请求数/字节、峰值内存、可见三角形、失败请求和截图；
6. 先报告参数差异，再报告性能与视觉结果，不把不同 profile 的结果合并成一个优劣结论。

这个实验能区分三类原因：Cesium loader 参数、浏览器/application bundle、publication 本身的
HLOD/纹理结构。只有第三类才应回流到 mesh 生成算法。

### 10.1 已完成的共享配置探针

在 1200x760、同一 overview 相机、同一 shared options 下，Terra 与 Remesh 的运行时配置逐项相等；
统一 SSE 滑块从 8 改为 4 时，两边同步变为 4。SSE 8/16/32 三次 overview 探针均无请求失败，但
Remesh 都请求约 271 个 GLB（约 114 MB），并在 30 秒时仍有内容处理中。该结果证明控制变量已经
统一，同时指出下一步速度瓶颈位于 publication 结构，而不是左右 Viewer 参数不一致。
