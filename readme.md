# 构建稀疏冠脉标签投影数据集

本仓库提供一条两阶段的数据生成流水线：

1. 先把原始冠脉分割标签裁剪、重采样到统一的局部体素空间。
2. 再基于重采样后的冠脉标签和原始 CT 体数据，生成指定角度下的 2D DRR 投影、Mesh 轮廓、深度图和对齐点云。

## 总体流程

```text
原始 coronary 分割 NIfTI
    -> crop 子命令：分离 LCA / RCA
    -> crop 子命令：裁剪 ROI、翻转正向 spacing、重采样
    -> 得到局部冠脉标签 NIfTI
    -> project 子命令：读取原始 volume / coronary 和重采样标签
    -> project 子命令：根据自定义 (alpha, beta) 角度构造 cone-beam 几何与 PyTorch3D 渲染
    -> 保存 DRR、mask、depth、point cloud 和可视化结果
```

## 1. crop 子命令

- [x] ~~**现在 `crop` 处理尺寸存在问题 比如即使设定了 `target-shape`，输出的 shape 仍然可能不完全等于目标值，可能是因为 spacing 的优先级更高。有一些长度超过 target shape 的边不会进行处理**~~ **已修复：`target_spacing` 和 `target_shape` 现在互斥，二者只能选其一。若设定了 `target_spacing`，则只按 spacing 缩放，忽略 `target_shape`；否则按 `target_shape` 缩放。**

`crop` 的职责只有一个：把原始冠脉标签整理成统一尺寸、统一 spacing、统一方向的局部 ROI。核心逻辑都在包内以下模块中：

- `[sparse_view_dataset/preprocess.py](sparse_view_dataset/preprocess.py)`：冠脉分离、ROI 计算、裁剪与重采样的主流程。
- `[sparse_view_dataset/affine_transforms.py](sparse_view_dataset/affine_transforms.py)`：affine 修正、spacing 翻转与中心化等变换工具.

### 运行原理

1. 使用连通域把冠脉分成两个主分支，并按质心位置区分 LCA 和 RCA。
2. 对每个分支计算轴对齐包围盒，并向外扩展若干体素。
3. 如果 affine 中存在负 spacing，则翻转数据并同步修正 affine，统一空间方向。
4. 将裁剪后的 ROI 重采样——`target_spacing` 和 `target_shape` 互斥，只能选其一：
   - 如果指定 `target-spacing`，则只按 spacing 缩放，`target-shape` 被忽略；
   - 否则按 `target-shape` 缩放。

### 使用方法

当前主流程如下：

```bash
python main.py crop ./ori_data/asoca/coronary/ data/asoca_size128_spacing0-7 --target-spacing 0.7 --target-shape 128 128 128
```

也可以直接使用包入口：

```bash
python -m sparse_view_dataset crop ./ori_data/asoca/coronary/ data/asoca_size128_spacing0-7 --target-spacing 0.7 --target-shape 128 128 128
```

参数含义：

- 输入目录：`./ori_data/asoca/coronary/`，需要包含原始冠脉标签 NIfTI 文件。
- 输出目录：`data/asoca_size128_spacing0-7`
- 目标 spacing：`0.7`
- 目标 shape：`128 128 128`

### 输出结果

如果输入文件名是 `Diseased_17.nii.gz`，输出通常会是：

```text
data/asoca_size128_spacing0-7/
    Diseased_17/
        Diseased_17_lca.nii.gz
        Diseased_17_rca.nii.gz
```

如果开启 `--saving-pt`，还会同时保存对应的 `.pt` 文件。

## 2. project 子命令

`project` 的职责是把重采样后的冠脉标签和原始 CT 体数据转成投影训练样本。核心逻辑在包内以下模块中：

- `[sparse_view_dataset/projection.py](sparse_view_dataset/projection.py)`：投影流程的总控与文件级别的批处理。
- `[sparse_view_dataset/cone_beam.py](sparse_view_dataset/cone_beam.py)`：投影算子封装 — 使用 DiffDRR（已移除 ODL 兼容路径）。
- `[sparse_view_dataset/torch3d_render.py](sparse_view_dataset/torch3d_render.py)`：基于 PyTorch3D 的 mask/depth 渲染器与相机设置。
- `[sparse_view_dataset/mesh_utils.py](sparse_view_dataset/mesh_utils.py)`：从体数据提取 mesh、平滑等工具（PyVista / marching-cubes 相关）。
- `[sparse_view_dataset/visualize.py](sparse_view_dataset/visualize.py)`：可视化、GIF 生成与导出小工具。

### 运行原理

1. 读取重采样后的冠脉标签，以及原始 coronary 和 volume。
2. 重新分离原始 coronary 中的 LCA / RCA，并选出当前分支。
3. 将原始体数据转换为模拟衰减系数，其中冠脉区域设置为更高的碘造影剂衰减值，背景近似水，无效值置零。
4. 把重采样后的冠脉标签中心化，让局部 ROI 的中心与世界坐标原点对齐，便于投影几何统一。
5. 同时将原始体数据和冠脉标签数据的 affine 修正为同一套坐标约定（世界坐标中心在重采样后的冠脉ROI中心），保证后续 DiffDRR 和 PyTorch3D 的空间对齐。
6. 使用 DiffDRR 生成指定 (alpha, beta) 角度下的 DRR（ZXY 欧拉角，intrinsic）。角度可通过命令行直接指定或随机生成。
7. 使用 PyVista 提取冠脉表面 mesh，再通过 PyTorch3D 渲染出 mask 和 depth。
8. 对冠脉体素点和中心线点进行同样的空间对齐，并输出成点云。

### 使用方法

当前主流程如下：

```bash
# 直接指定 alpha/beta 角度对
python main.py project data/asoca_size128_spacing0-7/ ./ori_data/asoca/ ./data/asoca_proj_128 \
    --proj-size 128 128 --alpha-betas 30 45 --alpha-betas 1.2 4.7  --vis

# 随机角度测试
python main.py project data/asoca_size128_spacing0-7/ ./ori_data/asoca/ ./data/asoca_proj_128 \
    --proj-size 128 128 --num-random 32
```

也可以直接使用包入口：

```bash
python -m sparse_view_dataset project data/asoca_size128_spacing0-7/ ./ori_data/asoca/ ./data/asoca_proj_128 --proj-size 128 128 --num-random 32
```

参数含义：

- 重采样冠脉目录：`data/asoca_size128_spacing0-7/`
- 原始数据目录：`./ori_data/asoca/`
- 输出目录：`./data/asoca_proj_128`
- 投影尺寸：`128 128`
- `--alpha-betas`：直接指定角度对（度），格式为 `--alpha-betas 30 45 --alpha-betas 1.2 4.7`
- `--num-random`：随机生成 N 个 (alpha, beta) 角度对用于测试
- `--vis`：是否生成可视化图像
- `--num-workers`：并行处理进程数（默认 4）
- `-d / --device`：指定 CUDA 设备 ID，可重复（如 `-d 0 -d 1`）

**C-arm 几何参数**（影响投影图像的空间分辨率与放大倍率）：
- `--dde`：探测器到世界原点距离（mm，默认 400）。增大 → 探测器更远 → 图像放大倍率增大。
- `--dso`：射线源到世界原点距离（mm，默认 1400）。增大 → 源更远 → 图像放大倍率减小。
- `--det-spacing`：探测器像素间距（mm/pixel，默认 0.3）。减小 → 像素更精细 → 视野变小。

### 输出结果

每个病例会在对应角度配置目录下生成一个 `.pt` 文件和可视化目录，结构类似：

```text
data/asoca_proj_128/
    random_32_projs/                          # 角度配置名
        Diseased_17_lca.pt
        Diseased_17_rca.pt
        vis/
            Diseased_17_lca/
                projs.gif
                depth.gif
                mask_2d.gif
                bg_mask_and_projs.gif
                cl_mask_and_depth.gif
    custom_5_projs/                           # 另一个角度配置
        Diseased_17_lca.pt
        Diseased_17_rca.pt
```

`.pt` 文件中包含以下键：

- `projs`：DRR 投影（取负值后归一化，从密度投影近似转换为X射线强度，数值越低表示X射线强度低，对应密度越大），shape 为 `(num_projs, H, W)`。
- `label_projs`：冠脉标签的密度投影（已归一化），shape 为 `(num_projs, H, W)`
- `mask_2d`：mesh 投影得到的二维二值轮廓，shape 为 `(num_projs, H, W)`。
- `depth`：mesh 投影得到的深度图，shape 为 `(num_projs, H, W)`, 值为 mesh 表面到相机的距离。
- `bg_mask`：冠脉前景体素点云，shape 为 `(num_projs, N, 3)`。
- `cl_mask`：中心线点云，shape 为 `(num_projs, N, 3)`。

其中点云坐标已经和二维投影对齐，可以把 `x`、`z` 看作图像平面中的位置，把 `y` 看作深度或高度通道，但由于 NDC 空间归一化，无法与 `depth` 完全对应，实际训练时仍然需要使用 `depth`。由于 `depth` 来自 mesh 表面而不是体素内部，因此它和点云只能近似对应，不能要求逐点完全相等。

## 3. project-once 子命令

`project-once` 用于对**单个病例**快速生成投影，无需批处理目录结构。适合调试、测试单组参数或处理真实临床数据。

### 运行原理

与 `project` 的核心投影逻辑相同（共用 `_project_one_case_inner`），区别在于：

- 直接指定冠脉标签 NIfTI 和原始 CT 体积 NIfTI 的路径。
- 可通过 `--ref-dicom-or-json` 从 DICOM 文件（`.dcm`）或 JSON 文件（`.json`）中**自动读取**投影参数：
  - 图像尺寸（`Rows`, `Columns`） → `proj_size`
  - 机架角度（`PositionerPrimaryAngle`, `PositionerSecondaryAngle`） → `alpha`, `beta`
  - 源到探测器距离（`DistanceSourceToDetector`）和源到患者距离（`DistanceSourceToPatient`） → 自动计算 `dde` 和 `dso`
  - 探测器像素间距（`ImagerPixelSpacing`） → `det_spacing`
- 任何 CLI 直接传入的参数（`--proj-size`, `--alpha-betas`, `--dde`, `--dso`, `--det-spacing`）**会覆盖**参考文件中读取的值。

### 使用方法

```bash
# 从 DICOM 文件读取所有参数（proj_size、角度、dde/dso、det_spacing）
python main.py project-once \
    data/asoca_size128_spacing0-7/Diseased_17/Diseased_17_lca.nii.gz \
    ori_data/asoca/volume/Diseased_17.nii.gz \
    output/single_case/ \
    --ref-dicom-or-json ori_data/ref.json \
    --vis

# 从 JSON 读取基础参数，但覆盖角度和 dde/dso
python main.py project-once \
    data/asoca_size128_spacing0-7/Diseased_17/Diseased_17_lca.nii.gz \
    ori_data/asoca/volume/Diseased_17.nii.gz \
    output/single_case/ \
    --ref-dicom-or-json ori_data/ref.json \
    --alpha-betas 30 45 \
    --dde 261.87 --dso 735.13 \
    --vis

# 完全手动指定所有参数
python main.py project-once \
    data/asoca_size128_spacing0-7/Diseased_17/Diseased_17_lca.nii.gz \
    ori_data/asoca/volume/Diseased_17.nii.gz \
    output/single_case/ \
    --proj-size 512 512 \
    --alpha-betas 39.2 0.1 \
    --dde 261.87 --dso 735.13 \
    --det-spacing 0.278875 \
    --vis
```

参数含义：

- `coronary_path`：重采样后的冠脉标签 NIfTI 文件路径（例如 `.../Diseased_17_lca.nii.gz`）。
- `volume_path`：原始 CT 体积 NIfTI 文件路径。
- `output_dir`：输出目录，结果保存为 `{case_name}_{branch_type}.pt`。
- `--ref-dicom-or-json`：参考 DICOM（`.dcm`）或 JSON（`.json`）文件，从中读取投影参数。
- `--proj-size`：覆盖投影图像尺寸（rows, cols）。
- `--alpha-betas`：覆盖角度对（度），格式为 `--alpha-betas 39.2 0.1`。
- `--dde`：覆盖探测器到世界原点距离（mm）。
- `--dso`：覆盖射线源到世界原点距离（mm）。
- `--det-spacing`：覆盖探测器像素间距（mm/pixel）。
- `--vis`：是否生成可视化图像。

### 输出结果

与 `project` 子命令的输出格式一致，`.pt` 文件包含 `projs`、`label_projs`、`mask_2d`、`depth`、`bg_mask`、`cl_mask` 等键（详见上一节）。

## 4. 数据与坐标约定

这个项目最重要的约定是“先统一空间，再做投影”。具体来说：

- 先通过裁剪和重采样，把不同病例归一到相似的局部体素空间。
- 再通过 affine 修正、中心化和重排，让三维体数据、mesh、点云和二维投影使用同一套几何约定。

这也是为什么代码里会同时处理数据数组和 affine 矩阵。只改数组不改 affine，会导致后续 DiffDRR、PyTorch3D 和点云坐标错位。
### 图像坐标约定

投影图像在不同模块中遵循不同的坐标约定，需要在可视化时做相应旋转对齐：

| 模块/库 | 原点位置 | 行方向 (dim=1) | 列方向 (dim=2) | 约定 |
|:---|:---|:---|:---|:---|
| DiffDRR / matplotlib / OpenCV | 左上角 (A) | 向下 (X) | 向右 (Y) | opencv 屏幕约定 |
| PyVista / VTK | 左下角 (B) | 向右 (X) | 向上 (Y) | 笛卡尔/OpenGL texture 约定 |
| PyTorch3D (NDC) | 左上角 (A) | 向下 | 向右 | opencv 屏幕约定 |

之前（ODL 时代）的坐标流转路径：
1. PyTorch3D (A) → ODL 投影 → 输出为 (B) 约定
2. ODL (B) → `save_gif(origin="lower", transpose)` → pyplot (A)

现在（DiffDRR 时代）的坐标流转路径：
1. DiffDRR (A), PyTorch3D (A) → 统一存储为 (A) 约定
2. 可视化 → `torch.rot90(projs, k=-1, dims=(1,2))` → PyVista/VTK (B) 约定

关键变更：
- **`torch.rot90(projs, k=-1, dims=(1, 2))`**：将图像逆时针旋转 90°，把 opencv 约定 (A) 转为 VTK 约定 (B)。对应 `visualize.py` 中 `plot_cloud_and_projs` 函数的首行操作。
- **去掉了 `save_gif` 中的 `origin="lower"` 和 `.transpose(-1, -2)`**：因为投影数据统一使用 (A) 约定存储，pyplot 默认的 `origin="upper"` 就是正确显示方式，不再需要坐标翻转。
- **去掉了 `torch3d_render.py` 中的 `.rot90(-1, [-2, -1])`**：之前因为 ODL 输出为 (B) 而 PyTorch3D 渲染为 (A)，需要旋转对齐；现在两者都是 (A)，不再需要旋转。

### 坐标系 (RAS)

所有几何计算使用 RAS 右手坐标系：

| 轴 | 方向 | 正值 | 对应人体方向 |
|:---|:---|:---|:---|
| X | Right | → 向右 | 患者右侧 |
| Y | Anterior | → 前方 | 患者前方（面朝上） |
| Z | Superior | → 向上 | 患者头侧 |

### 角度约定 (alpha / beta)

投影角度遵循 DICOM XA Positioner Module (C.8.7.5) 定义：

- [C.8.7.5 XA Positioner Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.7.5.html#sect_C.8.7.5.1.2)

使用 C-arm 风格的 (alpha, beta) 欧拉角对（轴顺序 Z-X, intrinsic）：

| 参数 | DICOM 名称 | 转轴 | 正向含义 |
|:---|:---|:---|:---|
| α (alpha) | Positioner Primary Angle | Z (Superior → Inferior) | 从右向前 → 绕 Z 轴 `+α` 旋转 |
| β (beta) | Positioner Secondary Angle | 旋转后的 X (Right → Left) | 从头侧向前 → 绕旋转后的 X 轴 `+β` 旋转 |

源位置计算公式：

```
src = R_z(alpha) @ R_x(beta) @ (0, dso, 0)
```

其中 `dso` 为源到世界原点的距离，初始源位置在患者前方 `(0, dso, 0)`。

### C-arm 几何参数（dde / dso / det_spacing）

这些参数共同控制 C-arm 的几何配置和投影图像的物理分辨率：

```
SDD (Source-Detector Distance) = dde + dso
SOD (Source-Object Distance)   = dso
```

| 参数 | 说明 | 默认值 | 增大后的效果 |
|:---|:---|:---|:---|
| `dde` | 探测器到世界原点距离 (mm) | 400 | 探测器远离 → 放大倍率增大 |
| `dso` | 射线源到世界原点距离 (mm) | 1400 | 源远离 → 放大倍率减小 |
| `det_spacing` | 探测器像素间距 (mm/pixel) | 0.3 | 像素变粗 → 视野增大，分辨率降低 |

从 DICOM 标签计算：
- `dso = DistanceSourceToPatient (0018,1111)`
- `dde = DistanceSourceToDetector (0018,1110) - DistanceSourceToPatient (0018,1111)`

## 5. 环境安装

推荐使用 pixi：

```bash
pixi install
```

如果要完全复刻环境，可以使用：

```bash
pixi install --frozen
```

如果使用 conda，可通过以下方式安装：

```bash
conda env create -f environment.yaml
```

## 6. 注意事项

- `project` 和 `project-once` 依赖 CUDA、DiffDRR、PyTorch3D 和 PyVista，运行前需要保证这些组件可用。
- 投影算子仅使用 DiffDRR（trilinear 渲染器），已移除 ODL 依赖。
- 原始数据目录需要同时包含 `coronary/` 和 `volume/` 两个子目录，且同一病例文件名必须一致。
- `separate_coronary` 默认按连通域和质心位置区分 LCA / RCA，假设输入坐标方向与原始数据一致。
- 如果输入标签不是标准 NIfTI，或者冠脉不是两个主要连通分支，分支拆分结果可能不稳定。

## 7. 开发建议

- 如果你要改裁剪、连通域或 affine 规则，优先看 [sparse_view_dataset/preprocess.py](sparse_view_dataset/preprocess.py) 和 [sparse_view_dataset/affine_transforms.py](sparse_view_dataset/affine_transforms.py)。
- 如果你要改投影几何或投影算子，优先看 [sparse_view_dataset/cone_beam.py](sparse_view_dataset/cone_beam.py)（ProjectionConeBeam — DiffDRR）以及 [sparse_view_dataset/projection.py](sparse_view_dataset/projection.py)（pipeline 编排）。
- 如果你要改 mesh 渲染或点云对齐，优先看 [sparse_view_dataset/torch3d_render.py](sparse_view_dataset/torch3d_render.py)。
- 如果你要改 mesh 相关实现，优先看 [sparse_view_dataset/mesh_utils.py](sparse_view_dataset/mesh_utils.py)。
- 如果你要改可视化输出或 GIF 生成，优先看 [sparse_view_dataset/visualize.py](sparse_view_dataset/visualize.py)。
- 如果你要改 I/O（NIfTI / .pt），优先看 [sparse_view_dataset/io.py](sparse_view_dataset/io.py)。
- 如果你要改命令行参数、增加子命令，优先看 [sparse_view_dataset/cli.py](sparse_view_dataset/cli.py)。

## 8. 命令速查

预处理：

```bash
python main.py crop ./ori_data/asoca/coronary/ data/asoca_size128_spacing0-7 --target-spacing 0.7 --target-shape 128 128 128
```

批量生成投影：
- 角度通过以下两种方式之一指定（二选一）：
  - `--alpha-betas 30 45 --alpha-betas 1.2 4.7`：命令行直接指定角度对（度）
  - `--num-random N`：随机生成 N 个角度用于测试
- `--vis`：是否生成可视化图像
- `--num-workers` 可以增加数据加载的并行度，默认为 4
- `--devices -d` 可以指定多个 GPU 进行并行处理，（例如 `-d 0 -d 1`） 默认为 0。每个 GPU 分配 num-workers / num-devices 个数据加载进程
- `--dde / --dso / --det-spacing`：C-arm 几何参数，详见上述说明


```bash
# 直接指定角度（度）
python main.py project data/asoca_size128_spacing0-7/ ./ori_data/asoca/ ./data/asoca_proj_128 \
    --proj-size 128 128 -d 0 -d 1 --alpha-betas 30 45 --alpha-betas 1.2 4.7 --vis --num-workers 16
```

单病例投影（从 DICOM/JSON 参考文件读取参数）：

```bash
python main.py project-once \
    data/asoca_size128_spacing0-7/Diseased_17/Diseased_17_lca.nii.gz \
    ori_data/asoca/volume/Diseased_17.nii.gz \
    output/single_case/ \
    --ref-dicom-or-json ori_data/ref.json \
    --vis
```

## 9. TODO

- [ ] 需要更新下载链接

可以在这里下载 [test_data](https://drive.google.com/drive/folders/1Gt5i_6Yvr-s1T9pTs_or9qTj4VUfyJ2L?usp=sharing)

所有测试文件位于 `tests/` 目录下，测试输出统一保存在 `tests/output/` 中。

运行全部测试：

```bash
pixi run python -m pytest tests/
```

### 测试文件说明

| 文件 | 内容 |
|:---|:---|
| `tests/test_conebeam_proj.py` | 基础 cone-beam 投影单元测试（需下载 test_data） |
| `tests/test_conebeam_alpha_beta.py` | (alpha, beta) 模式：源位置公式验证、前向投影、渲染器 |
| `tests/test_angle_configs.py` | CSV 加载、角度配置解析、CLI 参数解析 |
| `tests/test_integration_real_data.py` | 使用 `data/asoca_size128_spacing0-7` + `ori_data/asoca` 真实数据的集成测试 |
| `tests/test_projection.py` | 设备调度、失败日志等工具函数 |
| `tests/test_geometry.py` | affine 变换工具函数 |
| `tests/test_preprocess.py` | 预处理流程单元测试 |
| `tests/test_cli_smoke.py` | CLI 子命令注册冒烟测试 |