# 构建稀疏冠脉标签投影数据集

本仓库提供一条两阶段的数据生成流水线：

1. 先把原始冠脉分割标签裁剪、重采样到统一的局部体素空间。
2. 再基于重采样后的冠脉标签和原始 CT 体数据，生成 2D DRR 投影、Mesh 轮廓、深度图和对齐点云。

这两步分别由 [crop_resample.py](crop_resample.py) 和 [proj_odl.py](proj_odl.py) 完成。前者负责把原始三维标签整理成适合后续投影的标准输入，后者负责把三维标签和体数据转成可用于监督学习的二维投影样本。

## 总体流程

```text
原始 coronary 分割 NIfTI
    -> 分离 LCA / RCA
    -> 裁剪 ROI 并扩展边界
    -> 统一 affine 方向并重采样
    -> 得到局部冠脉标签 NIfTI
    -> 结合原始 volume / coronary
    -> 构造 ODL cone-beam 几何
    -> 生成 DRR、mask、depth、point cloud
    -> 保存为 .pt 和可视化结果
```

## 1. crop_resample.py

这个脚本的目标是把原始冠脉分割标签处理成统一尺寸、统一 spacing、统一方向的局部 ROI。它会先识别 LCA 和 RCA 两个分支，再分别裁剪、翻转、重采样，最后输出两个独立的 NIfTI 文件。

### 设计目的

原始冠脉标签通常体积较大，前景只占据很小一部分。如果直接对整个进行后续学习训练，计算开销高，而且不同病例之间的局部冠脉尺度和位置差异很大。这个阶段的设计目标是：

- 把有效前景缩到更小的 ROI，便于下游模型训练。
- 把所有病例归一到统一的体素尺寸和 spacing，便于批量训练和比较。
- 通过仿射矩阵修正，保持裁剪、翻转、重采样后的世界坐标一致性。

### 运行原理

脚本内部主要做了四件事：

1. 使用连通域把冠脉标签拆成两个大分支，并根据连通域质心位置区分 LCA 和 RCA。
2. 对每个分支计算轴对齐包围盒，并向外扩展若干体素，保留足够上下文。
3. 如果某个轴的 affine spacing 为负，则把数据沿该轴翻转，并同步修正 affine，保证空间方向统一。
4. 将裁剪后的 ROI 重采样到目标 shape；如果指定了 `target-spacing`，则会先按 spacing 缩放，再根据目标 shape 做必要填充。

### 使用方法

当前主流程如下：

```bash
python crop_resample.py ./ori_data/asoca/coronary/ data/asoca_size128_spacing0-7 --target-spacing 0.7 --target-shape 128 128 128
```

参数含义：

- 输入目录：`./ori_data/asoca/coronary/`，需要包含原始冠脉标签 NIfTI 文件。
- 输出目录：`data/asoca_size128_spacing0-7`
- 目标 spacing：`0.7`
- 目标 shape：`128 128 128`

### 输出结果

如果输入文件名是 `Diseased_17.nii.gz`，那么输出通常会是：

```text
data/asoca_size128_spacing0-7/
    Diseased_17/
        Diseased_17_lca.nii.gz
        Diseased_17_rca.nii.gz
```

如果开启 `--saving-pt`，还会同时保存对应的 `.pt` 文件。

## 2. proj_odl.py

这个脚本负责把重采样后的冠脉标签转成投影数据。它不是单纯做一个二维投影，而是同时生成三类互补监督信号：

- ODL 模拟得到的 DRR 投影 `projs`
- PyTorch3D 渲染得到的 2D 轮廓 `mask_2d` 和深度图 `depth`
- 与投影方向对齐的点云 `bg_mask` 和 `cl_mask`

### 设计目的

这一阶段的核心思路是把同一个三维对象从多个表示方式同时输出：

- `projs` 负责模拟真实 X 射线投影外观。
- `mask_2d` 和 `depth` 负责提供几何监督。
- `bg_mask` 和 `cl_mask` 负责保留点级别结构信息。

这样做的好处是，后续模型既能学到投影图像的外观分布，也能学到冠脉轮廓和空间深度关系。

### 运行原理

脚本大致分成以下步骤：

1. 读取重采样后的冠脉标签，以及原始 coronary 和 volume。
2. 重新分离原始 coronary 中的 LCA / RCA，并取出和当前文件对应的分支。
3. 将原始体数据转换为模拟衰减系数，其中冠脉区域使用更高的碘造影剂衰减值，背景近似水，无效填充值置零。
4. 把重采样后的冠脉标签中心化，让局部 ROI 的中心与世界坐标原点对齐，便于投影几何统一。
5. 同时将原始体数据和冠脉标签数据的 affine 修正为同一套坐标约定（世界坐标中心在重采样后的冠脉ROI中心），保证后续 ODL 和 PyTorch3D 的空间对齐。
6. 使用 ODL 构造 cone-beam 几何并生成多视角 DRR。
7. 使用 PyVista 提取冠脉表面 mesh，再通过 PyTorch3D 渲染出 mask 和 depth。
8. 对冠脉体素点和中心线点进行同样的空间对齐，并输出成点云。

### 使用方法

当前主流程如下：

```bash
python proj_odl.py data/asoca_size128_spacing0-7/ ./ori_data/asoca/ ./data/asoca_proj_128 --proj-size 128 128 --vis-num-projs 32
```

参数含义：

- 重采样冠脉目录：`data/asoca_size128_spacing0-7/`
- 原始数据目录：`./ori_data/asoca/`
- 输出目录：`./data/asoca_proj_128`
- 投影尺寸：`128 128`
- 可视化投影数量：`32`

### 输出结果

每个病例会在对应投影数量目录下生成一个 `.pt` 文件。默认结构类似：

```text
data/asoca_proj_128/
    32_projs/
        Diseased_17.pt
        ...
        vis/
            Diseased_17/
                projs.gif
                depth.gif
                mask_2d.gif
                bg_mask_and_projs.gif
                cl_mask_and_depth.gif
            ...
```

`.pt` 文件中包含以下键：

- `projs`：ODL 生成的 DRR 投影，shape 为 `(num_projs, H, W)`。
- `mask_2d`：mesh 投影得到的二维二值轮廓，shape 为 `(num_projs, H, W)`。
- `depth`：mesh 投影得到的深度图，shape 为 `(num_projs, H, W)`, 值为mesh表面到相机的距离。
- `bg_mask`：冠脉前景体素点云，shape 为 `(num_projs, N, 3)`。
- `cl_mask`：中心线点云，shape 为 `(num_projs, N, 3)`。

其中点云坐标已经和二维投影对齐，可以把 `x`、`z` 看作图像平面中的位置，把 `y` 看作深度或高度通道，但由于 NDC 空间归一化，无法与 `depth` 完全对应，实际训练时仍然需要使用 `depth`。由于 `depth` 来自 mesh 表面而不是体素内部，因此它和点云只能近似对应，不能要求逐点完全相等。

## 3. 数据与坐标约定

这个项目里最重要的约定是“先统一空间，再做投影”。具体来说：

- 先通过裁剪和重采样，把不同病例归一到相似的局部体素空间。
- 再通过 affine 修正、中心化和重排，让三维体数据、mesh、点云和二维投影使用同一套几何约定。

这也是为什么脚本里会同时处理数据数组和 affine 矩阵。只改数组不改 affine，会导致后续 ODL、PyTorch3D 和点云坐标全部错位。

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

- `proj_odl.py` 依赖 CUDA、ODL、ASTRA 和 PyTorch3D，运行前需要保证这些组件都可用。
- 原始数据目录需要同时包含 `coronary/` 和 `volume/` 两个子目录，且同一病例文件名必须一致。
- `separate_coronary` 会根据连通域和质心位置区分 LCA / RCA，默认假设当前坐标系方向与原始数据一致。
- 当前实现即使指定 `CUDA_VISIBLE_DEVICES`，仍可能占用少量 GPU 0 显存，这通常来自 PyTorch3D、OpenGL 或 ASTRA 的底层初始化。
- 如果输入标签不是标准 NIfTI，或者冠脉不是两个主要连通分支，分支拆分结果可能不稳定。

## 6. 命令速查

预处理：

```bash
python crop_resample.py ./ori_data/asoca/coronary/ data/asoca_size128_spacing0-7 --target-spacing 0.7 --target-shape 128 128 128
```

生成投影：

```bash
python proj_odl.py data/asoca_size128_spacing0-7/ ./ori_data/asoca/ ./data/asoca_proj_128 --proj-size 128 128 --vis-num-projs 32
```
