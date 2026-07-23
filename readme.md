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

- [ ] TODO **现在 `crop` 处理尺寸存在问题 比如即使设定了 `target-shape`，输出的 shape 仍然可能不完全等于目标值，可能是因为 spacing 的优先级更高。有一些长度超过 target shape 的边不会进行处理**

`crop` 的职责只有一个：把原始冠脉标签整理成统一尺寸、统一 spacing、统一方向的局部 ROI。核心逻辑都在包内以下模块中：

- `[sparse_view_dataset/preprocess.py](sparse_view_dataset/preprocess.py)`：冠脉分离、ROI 计算、裁剪与重采样的主流程。
- `[sparse_view_dataset/affine_transforms.py](sparse_view_dataset/affine_transforms.py)`：affine 修正、spacing 翻转与中心化等变换工具.

### 运行原理

1. 使用连通域把冠脉分成两个主分支，并按质心位置区分 LCA 和 RCA。
2. 对每个分支计算轴对齐包围盒，并向外扩展若干体素。
3. 如果 affine 中存在负 spacing，则翻转数据并同步修正 affine，统一空间方向。
4. 将裁剪后的 ROI 重采样到目标 shape；如果指定 `target-spacing`，则先按 spacing 缩放，再做必要填充。

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
- `[sparse_view_dataset/cone_beam.py](sparse_view_dataset/cone_beam.py)`：投影算子封装 — (alpha, beta) 模式使用 DiffDRR，旧模式使用 ODL 兼容路径。
- `[sparse_view_dataset/torch3d_render.py](sparse_view_dataset/torch3d_render.py)`：基于 PyTorch3D 的 mask/depth 渲染器与相机设置。
- `[sparse_view_dataset/mesh_utils.py](sparse_view_dataset/mesh_utils.py)`：从体数据提取 mesh、平滑等工具（PyVista / marching-cubes 相关）。
- `[sparse_view_dataset/visualize.py](sparse_view_dataset/visualize.py)`：可视化、GIF 生成与导出小工具。

### 运行原理

1. 读取重采样后的冠脉标签，以及原始 coronary 和 volume。
2. 重新分离原始 coronary 中的 LCA / RCA，并选出当前分支。
3. 将原始体数据转换为模拟衰减系数，其中冠脉区域设置为更高的碘造影剂衰减值，背景近似水，无效值置零。
4. 把重采样后的冠脉标签中心化，让局部 ROI 的中心与世界坐标原点对齐，便于投影几何统一。
5. 同时将原始体数据和冠脉标签数据的 affine 修正为同一套坐标约定（世界坐标中心在重采样后的冠脉ROI中心），保证后续 DiffDRR 和 PyTorch3D 的空间对齐。
6. 使用 DiffDRR 生成指定 (alpha, beta) 角度下的 DRR（ZXY 欧拉角，intrinsic）。角度可通过 CSV 文件、命令行直接指定或随机生成。
7. 使用 PyVista 提取冠脉表面 mesh，再通过 PyTorch3D 渲染出 mask 和 depth。
8. 对冠脉体素点和中心线点进行同样的空间对齐，并输出成点云。

### 使用方法

当前主流程如下：

```bash
# 从 CSV 文件读取角度
python main.py project data/asoca_size128_spacing0-7/ ./ori_data/asoca/ ./data/asoca_proj_128 \
    --proj-size 128 128 --angle-csv angles.csv --num-of-vis 5

# 直接指定 alpha/beta 角度对
python main.py project data/asoca_size128_spacing0-7/ ./ori_data/asoca/ ./data/asoca_proj_128 \
    --proj-size 128 128 --alpha-betas "0.0,0.0;0.5,0.1;1.0,0.2" --num-of-vis 3

# 随机角度测试
python main.py project data/asoca_size128_spacing0-7/ ./ori_data/asoca/ ./data/asoca_proj_128 \
    --proj-size 128 128 --num-random 32

# 多个角度配置 + 多 GPU
python main.py project data/asoca_size128_spacing0-7/ ./ori_data/asoca/ ./data/asoca_proj_128 \
    --proj-size 128 128 --angle-csv angles_a.csv --angle-csv angles_b.csv \
    -d 0 -d 1 --num-workers 16 --num-of-vis 5
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
- `--angle-csv`：CSV 文件路径，含两列 (alpha, beta)，可重复指定多个文件
- `--alpha-betas`：直接指定角度对，格式为 `"alpha1,beta1;alpha2,beta2;..."`
- `--num-random`：随机生成 N 个 (alpha, beta) 角度对用于测试
- `--num-of-vis`：可视化前 N 个角度的结果（0 = 不可视化）
- `--num-workers`：并行处理进程数（默认 4）
- `-d / --device`：指定 CUDA 设备 ID，可重复（如 `-d 0 -d 1`）

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
        ...

`.pt` 文件中包含以下键：

- `projs`：DRR 投影（取负值后归一化，从密度投影近似转换为X射线强度，数值越低表示X射线强度低，对应密度越大），shape 为 `(num_projs, H, W)`。
- `label_projs`：冠脉标签的密度投影（已归一化），shape 为 `(num_projs, H, W)`
- `mask_2d`：mesh 投影得到的二维二值轮廓，shape 为 `(num_projs, H, W)`。
- `depth`：mesh 投影得到的深度图，shape 为 `(num_projs, H, W)`, 值为 mesh 表面到相机的距离。
- `bg_mask`：冠脉前景体素点云，shape 为 `(num_projs, N, 3)`。
- `cl_mask`：中心线点云，shape 为 `(num_projs, N, 3)`。

其中点云坐标已经和二维投影对齐，可以把 `x`、`z` 看作图像平面中的位置，把 `y` 看作深度或高度通道，但由于 NDC 空间归一化，无法与 `depth` 完全对应，实际训练时仍然需要使用 `depth`。由于 `depth` 来自 mesh 表面而不是体素内部，因此它和点云只能近似对应，不能要求逐点完全相等。

## 3. 数据与坐标约定

这个项目最重要的约定是“先统一空间，再做投影”。具体来说：

- 先通过裁剪和重采样，把不同病例归一到相似的局部体素空间。
- 再通过 affine 修正、中心化和重排，让三维体数据、mesh、点云和二维投影使用同一套几何约定。

这也是为什么代码里会同时处理数据数组和 affine 矩阵。只改数组不改 affine，会导致后续 ODL、PyTorch3D 和点云坐标错位。
### 坐标系 (RAS)

所有几何计算使用 RAS 右手坐标系：

| 轴 | 方向 | 正值 | 对应人体方向 |
|:---|:---|:---|:---|
| X | Right | → 向右 | 患者右侧 |
| Y | Anterior | → 前方 | 患者前方（面朝上） |
| Z | Superior | → 向上 | 患者头侧 |

### 角度约定 (alpha / beta)

投影角度使用 C-arm 风格的 (alpha, beta) 欧拉角对（轴顺序 Z-X, intrinsic）：

| 参数 | 名称 | 转轴 | 正向含义 |
|:---|:---|:---|:---|
| α (alpha) | RAO (Right Anterior Oblique) | Z (Superior) | 从前向右 = 绕 Z 轴 `-α` 旋转 |
| β (beta) | CRA (Cranial) | 旋转后的 X | 从前向下 = 绕 X 轴 `-β` 旋转 |

源位置计算公式：

```
src = R_z(-alpha) @ R_x(-beta) @ (0, -dso, 0)
```

其中 `dso` 为源到世界原点的距离，初始源位置在患者后方 `(0, -dso, 0)`。
## 4. 环境安装

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

## 5. 注意事项

- `project` 依赖 CUDA、DiffDRR、PyTorch3D 和 PyVista，运行前需要保证这些组件可用。
- 旧模式（无 alpha/beta 角度）仍使用 ODL RayTransform；新模式使用 DiffDRR（trilinear 渲染器）。
- 原始数据目录需要同时包含 `coronary/` 和 `volume/` 两个子目录，且同一病例文件名必须一致。
- `separate_coronary` 默认按连通域和质心位置区分 LCA / RCA，假设输入坐标方向与原始数据一致。
- 当前实现即使指定 `CUDA_VISIBLE_DEVICES`，仍可能占用少量 GPU 0 显存，这通常来自 PyTorch3D、OpenGL 或 ODL 的底层初始化。
- 如果输入标签不是标准 NIfTI，或者冠脉不是两个主要连通分支，分支拆分结果可能不稳定。

## 6. 开发建议

- 如果你要改裁剪、连通域或 affine 规则，优先看 [sparse_view_dataset/preprocess.py](sparse_view_dataset/preprocess.py) 和 [sparse_view_dataset/affine_transforms.py](sparse_view_dataset/affine_transforms.py)。
- 如果你要改投影几何或投影算子，优先看 [sparse_view_dataset/cone_beam.py](sparse_view_dataset/cone_beam.py)（ProjectionConeBeam — DiffDRR / ODL）以及 [sparse_view_dataset/projection.py](sparse_view_dataset/projection.py)（pipeline 编排）。
- 如果你要改 mesh 渲染或点云对齐，优先看 [sparse_view_dataset/torch3d_render.py](sparse_view_dataset/torch3d_render.py)。
- 如果你要改 mesh 相关实现，优先看 [sparse_view_dataset/mesh_utils.py](sparse_view_dataset/mesh_utils.py)。
- 如果你要改可视化输出或 GIF 生成，优先看 [sparse_view_dataset/visualize.py](sparse_view_dataset/visualize.py)。
- 如果你要改 I/O（NIfTI / .pt），优先看 [sparse_view_dataset/io.py](sparse_view_dataset/io.py)。
- 如果你要改命令行参数、增加子命令，优先看 [sparse_view_dataset/cli.py](sparse_view_dataset/cli.py)。

## 7. 命令速查

预处理：

```bash
python main.py crop ./ori_data/asoca/coronary/ data/asoca_size128_spacing0-7 --target-spacing 0.7 --target-shape 128 128 128
```

生成投影（CSV 中的角度以 **度** 为单位）：
- 角度可通过以下三种方式之一指定（三选一）：
  - `--angle-csv path.csv`：从 CSV 文件读取 (alpha, beta) 列，可重复指定多个文件
  - `--alpha-betas "a1,b1;a2,b2;..."`：命令行直接指定角度对（度）
  - `--num-random N`：随机生成 N 个角度用于测试
- `--num-of-vis N` 控制前 N 个角度的可视化（默认 0 = 不可视化）
- `--num-workers` 可以增加数据加载的并行度，默认为 4
- `--devices -d` 可以指定多个 GPU 进行并行处理，（例如 `-d 0 -d 1`） 默认为 0。每个GPU 分配 num-workers / num-devices 个数据加载进程


```bash
# CSV 文件方式（角度制）
python main.py project data/asoca_size128_spacing0-7/ ./ori_data/asoca/ ./data/asoca_proj_128 \
    --proj-size 128 128 -d 0 -d 1 --angle-csv sample_angles.csv --num-of-vis 32 --num-workers 16

# 直接指定角度（度）
python main.py project data/asoca_size128_spacing0-7/ ./ori_data/asoca/ ./data/asoca_proj_128 \
    --proj-size 128 128 -d 0 -d 1 --alpha-betas "0,0;10,5;20,10;30,5" --num-of-vis 4 --num-workers 16
```

## 8. test

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