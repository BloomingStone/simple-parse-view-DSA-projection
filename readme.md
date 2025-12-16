# 构建稀疏冠脉标签投影数据集

## 代码文件

### `crop_resample.py`

根据所给二值冠脉标签nii文件，先裁剪前景ROI，后缩放到指定 shape （和可能的给定 spacing）. 

裁剪并重新采样冠状动脉数据流程：
1. 从输入路径加载 NIfTI 文件，如果输入的是目录，则会迭代搜索所有 nii 文件，并保留文件结构为后续存储所用
2. 分离冠状动脉 LCA 和 RCA 分支
3. 裁剪和扩大每个分支的 ROI
4. 翻转数据并调整仿射使间距为正
5. 重新采样到由`target_shape`指定的目标形状（如果设置了`target_spacing`，也通过缩放和必要的填充对其进行重新采样，因此形状可能比`target_shape`更大）
6. 保存结果NIfTI文件

详细使用方式可通过 `python crop_resample.py --help` 查看。

推荐运行设置为 `size = (320, 320, 320), spacing = 0.4`. 对应的运行代码为

```bash
python crop_resample.py <input_dir> data/nii_size320_spacing0-4 --target-spacing 0.4 --target-shape 320 320 320
```

### `proj_odl.py`

使用ODL获得冠脉label DRR投影 作为剪影后的DSA图像。同时使用pyvista提取 表面mesh 并通过 pytorch3D 投影获得深度图和剪影。投影输入数据应为 `crop_resample.py` 处理后的数据。

处理流程:
1. 从输入目录加载所有.nii.gz文件
2. 对于每个文件：
    1. 加载三维体数据并生成网格
    2. 生成多个2D投影（由num_projects指定）
    3. 保存投影和导出数据（深度图，掩模）
    4. 当num_projs=32时，可选保存可视化
3. 使用multiprocessing并行处理文件

详细使用方式可通过 `python proj_odl.py --help` 查看
示例生成结果可用 `pytest test_proj_odl.py` 查看

一个运行示例如下, 设置投影结果大小为 128x128, 投影数量为 16 和 32, 可视化32视角投影结果
```bash
python proj_odl.py ./data/nii_size320_spacing0-4/ ./data/nii_size320_spacing0-4__projs128/ --proj-size 128 128 --num-projs 16 32 --vis-num-projs 32 --num-workers 4
```

## 数据结果解释

所生成 pt 文件由以下键值对组成：
- projs: 对冠脉标签的 DRR 投影，shape 为 (num_projs, W, H)
- mask_2d: 标签提取 mesh 后投影得到的 2D 二值标签，shape 为 (num_projs, W, H), dtype 为 uint8
- depth: 标签提取 mesh 后投影得到的 2D 深度图，shape 为 (num_projs, W, H), dtype 为 float32. 深度取值范围为 [0, 1]，对应原始 3D 冠脉标签区域。深度值为 Y 轴方向到冠脉 mesh 表面的距离。
- bg_mask: 冠脉格点缩放旋转后得到的点云，旋转为与2D投影对齐，shape 为 (num_projs, N, 3)，取值范围为 [0, 1]
- cl_mask: sketonlize 得到 中心线格点后缩放旋转后得到的点云，旋转是为了与2D投影对齐，shape 为 (num_projs, N, 3), 取值范围为 [0, 1]

点云和depth的对应关系可参考如下
```
x y z = bg_mask[b, n]
depth[b, x*(W-1), z*(H-1)] ≈ y
```
注意是约等于，因为 depth 为 source 到最近的mesh**表面**的距离，而中心线和格点在mesh内部，两者存在偏差

## 环境安装

推荐使用pixi，可直接 `pixi install` 安装，如果要完全复刻环境，可使用 `pixi install --frozen` 安装 lock 文件定义的环境。

如果使用conda，可通过 environment.yml 安装环境：
```bash
conda env create -f environment.yml
```

**注意**：如需指定使用的 GPU device 需要同时指定 `CUDA_VISIBLE_DEVICES` 环境变量。如果使用pixi，可在 `.pixi/config.toml` 中调整以下内容：

```toml
[activation.env]
CUDA_VISIBLE_DEVICES = "3"
```

但目前不管如何设置，仍然会占用部分GPU0，大小约为 100MB * num_workders, 可能是 PyTorch3D / OpenGL / ASTRA 等库自行指定的，在GPU0显存已被占满时，可能无法正确运行。
