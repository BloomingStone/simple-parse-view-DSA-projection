# import sys
# import SimpleITK
# import numpy as np
import yaml
from skimage.metrics import structural_similarity as ssim
from xueming_data_process2 import *
import json
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import torch
# from torch.nn import functional as F
from cldice import soft_dice_cldice
import nibabel as nib
import time
import torch.backends.cudnn as cudnn
# from networks import Positional_Encoder, FFN, SIREN
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
cudnn.benchmark = True
# def visualize_pointcloud(points, title="Initial Gaussian Point Cloud",
#                          save_path="/media/I/xcw/3DGR-CAR-main/3dgs-car/fbp_train_vis"):
#     """
#     points: torch.Tensor or np.ndarray, shape (N, 3)
#     """
#     import numpy as np
#     import torch
#     import matplotlib.pyplot as plt

#     # 转 numpy
#     if isinstance(points, torch.Tensor):
#         points = points.detach().cpu().numpy()

#     fig = plt.figure(figsize=(7, 7))
#     ax = fig.add_subplot(111, projection='3d')

#     x, y, z = points[:, 0], points[:, 1], points[:, 2]

#     # 红色点云
#     ax.scatter(x, y, z, s=1, c='red')

#     # 比例一致
#     max_range = np.array([
#         (x.max()-x.min()),
#         (y.max()-y.min()),
#         (z.max()-z.min())
#     ]).max() / 2.0

#     mid = np.array([x.mean(), y.mean(), z.mean()])
#     ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
#     ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
#     ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

#     ax.set_title(title)

#     # 移除坐标轴
#     ax.set_axis_off()

#     # 保存但不显示
#     if save_path:
#         fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
#         print(f"Saved to {save_path}")

#     plt.close(fig)

import matplotlib.pyplot as plt
import imageio

def create_rotating_pointcloud_gif(xyz, density=None, scaling=None,
                                   save_path="initial_gaussians.gif",
                                   n_frames=36, point_size=1):
    """
    xyz: torch.Tensor or np.ndarray, shape [N,3]
    density: torch.Tensor or np.ndarray, shape [N,1] or None -> 控制颜色
    scaling: torch.Tensor or np.ndarray, shape [N,3] or None -> 控制点大小
    save_path: 输出 GIF 路径
    n_frames: 总帧数
    point_size: 点大小
    """
    # 转 numpy
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    N = xyz.shape[0]

    if density is not None and isinstance(density, torch.Tensor):
        density = density.detach().cpu().numpy().reshape(-1)
        d_norm = (density - density.min()) / (density.max() - density.min() + 1e-6)
        colors = np.zeros((N,3))
        colors[:,0] = d_norm          # R
        colors[:,2] = 1 - d_norm      # B
    else:
        colors = np.ones((N,3)) * 0.5

    if scaling is not None and isinstance(scaling, torch.Tensor):
        scaling = scaling.detach().cpu().numpy()
        sizes = np.linalg.norm(scaling, axis=1) * 10
    else:
        sizes = np.ones(N) * point_size

    os.makedirs("temp_frames", exist_ok=True)
    frame_paths = []

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()/2.0
    mid = np.array([x.mean(), y.mean(), z.mean()])

    for i in range(n_frames):
        ax.clear()
        ax.scatter(x, y, z, c=colors, s=sizes)
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
        ax.view_init(elev=30, azim=i*360/n_frames)
        plt.tight_layout()
        frame_path = f"temp_frames/frame_{i:03d}.png"
        plt.savefig(frame_path, dpi=150)
        frame_paths.append(frame_path)

    plt.close(fig)

    # 保存 GIF，loop=0 表示无限循环
    images = [imageio.imread(fp) for fp in frame_paths]
    imageio.mimsave(save_path, images, duration=0.1, loop=0)

    # 清理临时文件
    for fp in frame_paths:
        os.remove(fp)
    os.rmdir("temp_frames")

    print(f"[✔] Rotating Gaussian point cloud GIF saved → {save_path} (infinite loop)")

class PointCloudVisualizer1:
    @staticmethod
    def save_static_pointcloud(xyz, density=None, scaling=None,
                               save_path="pointcloud.png",
                               point_size=1,
                               view_elev=30,
                               view_azim=-45):
        """
        保存静态点云图
        xyz: torch.Tensor or np.ndarray, shape [N,3]
        density: 控制颜色, shape [N] 或 None
        scaling: 控制点大小, shape [N,3] 或 None
        save_path: 输出图片路径
        point_size: 默认点大小
        view_elev / view_azim: 视角
        """

        # 转 numpy
        if isinstance(xyz, torch.Tensor):
            xyz = xyz.detach().cpu().numpy()
        N = xyz.shape[0]

        # 颜色处理
        if density is not None:
            if isinstance(density, torch.Tensor):
                density = density.detach().cpu().numpy().reshape(-1)
            d_norm = (density - density.min()) / (density.max() - density.min() + 1e-6)
            colors = np.zeros((N, 3))
            colors[:,2] = d_norm      # R
            colors[:,0] = 1 - d_norm  # B
        else:
            colors = np.ones((N, 3)) * 0.5

        # 点大小处理
        if scaling is not None:
            if isinstance(scaling, torch.Tensor):
                scaling = scaling.detach().cpu().numpy()
            sizes = np.linalg.norm(scaling, axis=1) * 10
        else:
            sizes = np.ones(N) * point_size

        # 绘制静态点云
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = xyz[:,2], xyz[:,1], xyz[:,0]

        # 设置比例一致
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()/2.0
        mid = np.array([x.mean(), y.mean(), z.mean()])
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

        # 绘制点云
        ax.scatter(x, y, z, c=colors, s=sizes)

        # 固定视角
        ax.view_init(elev=view_elev, azim=view_azim)

        # 不显示坐标轴和网格
        ax.set_axis_off()
        ax.grid(False)

        # 保存图片
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f"[✔] Point cloud saved → {save_path}")
        return save_path

class PointCloudVisualizer2:
    @staticmethod
    def save_static_pointcloud(xyz, density=None, scaling=None,
                               save_path="pointcloud.png",
                               point_size=3,
                               view_elev=30,
                               view_azim=-45):
        """
        保存静态点云图
        xyz: torch.Tensor or np.ndarray, shape [N,3]
        density: 控制颜色, shape [N] 或 None
        scaling: 控制点大小, shape [N,3] 或 None
        save_path: 输出图片路径
        point_size: 默认点大小
        view_elev / view_azim: 视角
        """

        # 转 numpy
        if isinstance(xyz, torch.Tensor):
            xyz = xyz.detach().cpu().numpy()
        N = xyz.shape[0]

        # 颜色处理
        if density is not None:
            if isinstance(density, torch.Tensor):
                density = density.detach().cpu().numpy().reshape(-1)
            d_norm = (density - density.min()) / (density.max() - density.min() + 1e-6)
            colors = np.zeros((N, 3))
            colors[:,0] = d_norm      # R
            colors[:,2] = 1 - d_norm  # B
        else:
            colors = np.tile(np.array([[0, 0, 1]]), (N, 1))

        # 点大小处理
        if scaling is not None:
            if isinstance(scaling, torch.Tensor):
                scaling = scaling.detach().cpu().numpy()
            sizes = np.linalg.norm(scaling, axis=1) * 10
        else:
            sizes = np.ones(N) * point_size

        # 绘制静态点云
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2] #012,021,120,102,201

        # 设置比例一致
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()/2.0
        mid = np.array([x.mean(), y.mean(), z.mean()])
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

        # 绘制点云
        ax.scatter(x, y, z, c=colors, s=sizes)

        # 固定视角
        ax.view_init(elev=view_elev, azim=view_azim)

        # 不显示坐标轴和网格
        ax.set_axis_off()
        ax.grid(False)

        # 保存图片
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f"[✔] Point cloud saved → {save_path}")
        return save_path

import matplotlib.colors as mcolors
def save_depth_image(depth_map, cmap='viridis', rotate=0, flip=False, save_path="depth.png"):
    """
    保存平面深度图，深度为1的地方显示为白色
    
    参数:
        depth_map: torch.Tensor 或 np.ndarray, shape [H, W]
        cmap: matplotlib colormap，默认 'viridis'
        rotate: 旋转角度，可选 0/90/180/270
        flip: 是否上下翻转
        save_path: 输出图片路径
    """
    # 转 numpy
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()

    # 旋转
    if rotate != 0:
        depth_map = np.rot90(depth_map, k=rotate//90)
    
    # 上下翻转
    if flip:
        depth_map = depth_map[::-1, :]

    # 创建自定义 colormap，将深度为1的位置映射为白色
    base_cmap = plt.get_cmap(cmap)
    new_colors = base_cmap(np.linspace(0, 1, 256))
    new_colors[-1, :] = [1, 1, 1, 1]  # 将最大值对应白色
    custom_cmap = mcolors.ListedColormap(new_colors)

    # 归一化深度值到 [0,1]，并将 1 替换为最大值对应索引
    norm = mcolors.Normalize(vmin=depth_map.min(), vmax=1.0)

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存图片
    plt.figure(figsize=(6,6), facecolor='white')
    plt.imshow(depth_map, cmap=custom_cmap, norm=norm)
    plt.colorbar(label='Depth')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    print(f"[✔] Depth image saved → {save_path}")

# 定义一个函数来选择投影
def select_projections(projs, num_projections):
    total_projections = projs.size(1)
    step = total_projections // num_projections
    indices = [i * step for i in range(num_projections)]
    return projs[:,indices, :, :]
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true = y_true > 0
    y_pred = y_pred > 0
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

def psnr(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    return 20 * torch.log10(torch.max(y_true) / torch.sqrt(mse))

def psnr_mask(y_true, y_pred):
    gt_mask = y_true > 0
    y_true = y_true[gt_mask]
    y_pred = y_pred[gt_mask]
    mse = torch.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    max_pixel = 1
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def psnr_mask_projs(y_true, y_pred):
    gt_mask = y_true > 0
    y_true = y_true[gt_mask]
    y_pred = y_pred[gt_mask]
    mse = torch.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    max_pixel = 16
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def cldice_loss(y_true, y_pred, smooth=1e-6):
    y_true = y_true > 0
    y_pred = y_pred > 0
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum(y_pred) - tp
    fn = torch.sum(y_true) - tp
    soft_dice = (2*tp + smooth) / (2*tp + fp + fn + smooth)
    return 1 - soft_dice
#    s_cldice_loss = soft_dice_cldice(iter_=3, alpha=0.5, smooth=1.)
#    s_cldice = s_cldice_loss(pred_voxel, gt_voxel)
#indicators: dice, cldice, psnr, ssim, mse
def evaluate_voxel(pred_voxel, gt_voxel):
    '''
    pred_voxel: [1, x, y, z]
    gt_voxel: [1, x, y, z]
    '''
    pred_voxel = pred_voxel.squeeze(0)
    gt_voxel = gt_voxel.squeeze(0)

    dice = dice_coefficient(gt_voxel, pred_voxel)
    cldice = cldice_loss(gt_voxel, pred_voxel)
    psnr_val = psnr_mask(gt_voxel, pred_voxel)
    data_range = gt_voxel.max() - gt_voxel.min()
    # Determine the smaller side of the image
    smaller_side = min(gt_voxel.shape[-2], gt_voxel.shape[-1])

    # Ensure win_size is odd and less than or equal to the smaller side of the image
    win_size = smaller_side if smaller_side % 2 == 1 else smaller_side - 1

    # Calculate SSIM
    ssim_val = ssim(gt_voxel.cpu().numpy(), pred_voxel.cpu().numpy(), multichannel=True,data_range=data_range.cpu().item(), win_size=win_size)
    # ssim_val = ssim(gt_voxel.cpu().numpy(), pred_voxel.cpu().numpy(), multichannel = True, data_range=data_range.cpu().item())
    mse_val = mse(gt_voxel.cpu().numpy().flatten(), pred_voxel.cpu().numpy().flatten())

    return (dice.cpu().item(), cldice.cpu().item(), psnr_val.cpu().item(), ssim_val, mse_val)
#indictors: dice, cldice, psnr, ssim, mse
def evaluate_newprojs(pred_projs, gt_projs):
    '''
    :param pred_projs: [1, num_proj, x, y]
    :param gt_projs:   [1, num_proj, x, y]
    :return:
    '''
    pred_projs = pred_projs.squeeze(0)
    gt_projs = gt_projs.squeeze(0)

    dice = dice_coefficient(gt_projs, pred_projs)
    cldice = cldice_loss(gt_projs, pred_projs)
    psnr_val = psnr_mask_projs(gt_projs, pred_projs)
    data_range=gt_projs.max() - gt_projs.min()
    # Determine the smaller side of the image
    #如果ssim_val报错就设置为0
    # ssim_val = ssim(gt_projs.cpu().numpy(), pred_projs.cpu().numpy(), data_range=data_range.cpu().item())
    ssim_val = 0

    mse_val = mse(gt_projs.cpu().numpy().flatten(), pred_projs.cpu().numpy().flatten())
    return (dice.cpu().item(), cldice.cpu().item(), psnr_val.cpu().item(), ssim_val, mse_val)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.dice = []
        self.cldice = []
        self.psnr = []
        self.ssim = []
        self.mse = []

    def update(self, eval_res):
        self.dice.append(eval_res[0])
        self.cldice.append(eval_res[1])
        self.psnr.append(eval_res[2])
        self.ssim.append(eval_res[3])
        self.mse.append(eval_res[4])
    def average(self):
        return [np.mean(self.dice), np.mean(self.cldice), np.mean(self.psnr), np.mean(self.ssim), np.mean(self.mse)]
    # 将所有的评估指标和平均结果保存到csv文件中
    def save_to_csv(self, save_dir, filename):
        data = {'dice': self.dice, 'cldice': self.cldice, 'psnr': self.psnr, 'ssim': self.ssim, 'mse': self.mse}
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_dir, filename + '.csv'))



def evaluate_fbp(num_proj = 16, save_dir=r"/data/xuemingfu/NeRP/fbp_result"):
    algo = 'fbp'
        
    # projs_num = [2,4,8,16]
    
    num_proj = num_proj
    image_size = [128] * 3
    proj_size = [128] * 3
    ct_projector_train = ConeBeam3DProjector(image_size, proj_size, num_proj)
    ct_projector_new = ConeBeam3DProjector(image_size, proj_size, num_proj, start_angle=5)

    CCTA_eval = {'voxel': AverageMeter(),
                 'projs': AverageMeter()}
    CAS_eval = {'voxel': AverageMeter(),
                'projs': AverageMeter()}


    #Test CAS data
    # for filename in CAS_test_list:
    #     print("Start to evaluate " + str(filename) + " with " + str(num_proj) + " views")
    #     # volume data CAS
    #     data_path = os.path.join(CASDataset_path, str(filename) + '_volume.pt')
    #     gt_volume = torch.load(data_path)['volume'][0,:,:,:,:].cuda()  # [1, x, y, z]
    #     input_projs = ct_projector_train.forward_project(gt_volume)   #[1, num_proj, x, y]
    #     start = time.time()
    #     fbp_recon = ct_projector_train.backward_project(input_projs)
    #     voxel_result = evaluate_voxel(fbp_recon, gt_volume)
    #     CAS_eval['voxel'].update(voxel_result)
    #     end = time.time()
    #     print("num_views: ", num_proj)
    #     print("************Training time: s", end - start)
    #
    #     # save fbp_recon
    #     fbp_recon_saved = fbp_recon.squeeze(0).cpu().numpy()
    #     fbp_recon_saved = nib.Nifti1Image(fbp_recon_saved, np.eye(4))
    #     nib.save(fbp_recon_saved, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-fbp.nii.gz'))
    #
    #     # generate new projs
    #     new_projs_tr = ct_projector_new.forward_project(fbp_recon)
    #     new_projs_gt = ct_projector_new.forward_project(gt_volume)
    #     projs_eval = evaluate_newprojs(new_projs_tr, new_projs_gt)
    #     CAS_eval['projs'].update(projs_eval)
    #     # save new_projs to torch pt file
    #     torch.save(new_projs_tr, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-new_projs_train.pt'))
    #     torch.save(new_projs_gt, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-new_projs_label.pt'))

    # save evaluation results to csv, 并且通过评估结果的平均值的str字符串来命名文件
    # voxel_list = map(lambda x: "{:.3f}".format(x), CAS_eval['voxel'].average())
    # projs_list = map(lambda x: "{:.3f}".format(x), CAS_eval['projs'].average())
    # voxel_str = "-".join([str(number) for number in voxel_list])
    # projs_str = "-".join([str(number) for number in projs_list])
    # CAS_eval['voxel'].save_to_csv(save_dir, f"{algo}-CAS_voxel-views-{num_proj}-{voxel_str}")
    # CAS_eval['projs'].save_to_csv(save_dir, f"{algo}-CAS_projs-views-{num_proj}-{projs_str}")


    #Test CCTA data
    for filename in CCTA_test_list:
        print("Start to evaluate " + filename + " with " + str(num_proj) + " views")
        # volume data CCTA
        data_path = os.path.join(CCTADataset_path, filename + '_volume.pt')
        gt_volume = torch.load(data_path)['volume'][0,:,:,:,:].cuda()  # [1, x, y, z]
        input_projs = ct_projector_train.forward_project(gt_volume)   #[1, num_proj, x, y]

        #compare different reconstruction methods

        fbp_recon = ct_projector_train.backward_project(input_projs)
        voxel_result = evaluate_voxel(fbp_recon, gt_volume)
        CCTA_eval['voxel'].update(voxel_result)
        # save fbp_recon
        fbp_recon_saved = fbp_recon.squeeze(0).cpu().numpy()
        fbp_recon_saved = nib.Nifti1Image(fbp_recon_saved, np.eye(4))
        nib.save(fbp_recon_saved, os.path.join(save_dir, filename+ "-views-"+str(num_proj) + '-fbp.nii.gz'))

        #generate new projs
        new_projs_tr = ct_projector_new.forward_project(fbp_recon)
        new_projs_gt = ct_projector_new.forward_project(gt_volume)
        projs_eval = evaluate_newprojs(new_projs_tr, new_projs_gt)
        CCTA_eval['projs'].update(projs_eval)
        # save new_projs to torch pt file
        torch.save(new_projs_tr, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-new_projs_train.pt'))
        torch.save(new_projs_gt, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-new_projs_label.pt'))

    #save evaluation results to csv, 并且通过评估结果的平均值的str字符串来命名文件
    voxel_list = map(lambda x: "{:.3f}".format(x), CCTA_eval['voxel'].average())
    projs_list = map(lambda x: "{:.3f}".format(x), CCTA_eval['projs'].average())
    voxel_str = "-".join([str(number) for number in voxel_list])
    projs_str = "-".join([str(number) for number in projs_list])
    CCTA_eval['voxel'].save_to_csv(save_dir, f"{algo}-CCTA_voxel-views-{num_proj}-{voxel_str}")
    CCTA_eval['projs'].save_to_csv(save_dir, f"{algo}-CCTA_projs-views-{num_proj}-{projs_str}")

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)
import copy
def evaluate_nerp(num_proj=16, save_dir=r"/data/xuemingfu/NeRP/nerp_result", config_file=r"./configs/ct_recon_3d.yaml"):
    algo = 'nerp'
    # projs_num = [2,4,8,16]
    num_proj = num_proj
    image_size = [128] * 3
    proj_size = [128] * 3
    ct_projector_train = ConeBeam3DProjector(image_size, proj_size, num_proj)
    ct_projector_new = ConeBeam3DProjector(image_size, proj_size, num_proj, start_angle=5)

    CCTA_eval = {'voxel': AverageMeter(),
                 'projs': AverageMeter()}
    CAS_eval = {'voxel': AverageMeter(),
                'projs': AverageMeter()}
    config = get_config(config_file)
    # best_cldice 用于选择最好的cldice的模型  采用early stopping 机制, 深拷贝

    # Test CAS data
    for filename in CAS_test_list:
        print("Start to evaluate " + str(filename) + " with " + str(num_proj) + " views")
        best_psnr = 0
        patient = 0
        best_iter = 0
        # define nerp model
        # Setup input encoder:
        encoder = Positional_Encoder(config['encoder'])
        # Setup model
        if config['model'] == 'SIREN':
            model = SIREN(config['net'])
        elif config['model'] == 'FFN':
            model = FFN(config['net'])
        else:
            raise NotImplementedError
        model.cuda()
        model.train()

        # Load pretrain model
        pretrain = False
        if pretrain:
            model_path = os.path.join(config['pretrain_model_path'], str(filename) + "-views-" + str(num_proj) + '-model.pt')
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict['net'])
            encoder.B = state_dict['enc']
            print('Load pretrain model: {}'.format(model_path))
        # Setup optimizer
        if config['optimizer'] == 'Adam':
            optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']),weight_decay=config['weight_decay'])
        else:
            NotImplementedError
        # Setup loss function
        if config['loss'] == 'L2':
            loss_fn = torch.nn.MSELoss()
        elif config['loss'] == 'L1':
            loss_fn = torch.nn.L1Loss()
        else:
            NotImplementedError
        # volume data CAS  & test data  projs
        data_path = os.path.join(CASDataset_path, str(filename) + '_volume.pt')
        gt_volume = torch.load(data_path)['volume'][0, :, :, :, :].cuda()  # [1, x, y, z]
        input_projs = ct_projector_train.forward_project(gt_volume)  # [1, num_proj, x, y]
        # train nerp model
        # Input coordinates (x,y) grid and target image
        grid = create_grid_3d(*image_size)
        grid = grid.cuda()
        input_projs = input_projs.cuda()
        test_data = (grid, input_projs)  # [bs, z, x, y, 1]
        train_data = (grid, input_projs)  # [bs, n, h, w]
        scaler = GradScaler()
        max_iter = config['max_iter']
        starttime = time.time()
        for iterations in range(max_iter):
            model.train()
            optim.zero_grad()
            with autocast():
                train_embedding = encoder.embedding(train_data[0])
                train_output = model(train_embedding)
                train_projs = ct_projector_train.forward_project(train_output.transpose(0, 3))  # .squeeze(1))
                train_loss = 0.5 * loss_fn(train_projs, train_data[1])
            scaler.scale(train_loss).backward()
            scaler.step(optim)
            scaler.update()
            if iterations % 20 == 0:
                train_psnr = -10 * torch.log10(2 * train_loss).item()
                train_loss = train_loss.item()
                print("[Iteration: {}/{}] Train loss: {:.6g} | Train psnr: {:.6g}".format(iterations + 1, max_iter,train_loss,train_psnr))
            # 在循环或迭代结束后
            del train_embedding, train_output, train_loss
            torch.cuda.empty_cache()

            # Test nerp model
            # Compute testing psnr
            # if iterations == 0 or (iterations + 1) % 1 == 0:
            if iterations == 0 or (iterations + 1) % config['val_iter'] == 0:
                model.eval()
                with torch.no_grad():
                    test_embedding = encoder.embedding(test_data[0])
                    test_output = model(test_embedding)
                    # cldice = cldice_loss(gt_volume, test_output.transpose(0, 3))
                    test_projs = ct_projector_train.forward_project(test_output.transpose(0, 3))  # .squeeze(1))
                    test_loss = 0.5 * loss_fn(test_projs, test_data[1])
                    test_psnr = - 10 * torch.log10(2 * test_loss).item()

                    if test_psnr > best_psnr:
                        best_psnr = test_psnr
                        patient = 0
                        saved_model = copy.deepcopy(model.state_dict())
                        saved_encoder = copy.deepcopy(encoder.B)
                        best_iter = iterations
                        print("best_psnr: ", best_psnr)
                    else:
                        patient += 1
                        if patient > 6:
                            print("Early stopping at iteration: ", best_iter,"best_psnr: ", best_psnr)
                            break

        endtime = time.time()
        print("num_views: ", num_proj)
        print("************Training time: s", endtime - starttime)

        encoder.B = saved_encoder
        model.load_state_dict(saved_model)
        # save nerp_recon
        with torch.no_grad():
            test_embedding = encoder.embedding(test_data[0])
            test_output = model(test_embedding)
        fbp_recon = test_output.transpose(0, 3)
        voxel_result = evaluate_voxel(fbp_recon, gt_volume)
        CAS_eval['voxel'].update(voxel_result)
        # save fbp_recon
        fbp_recon_saved = fbp_recon.squeeze(0).cpu().numpy()
        fbp_recon_saved = nib.Nifti1Image(fbp_recon_saved, np.eye(4))
        nib.save(fbp_recon_saved, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-nerp.nii.gz'))

        # generate new projs
        new_projs_tr = ct_projector_new.forward_project(fbp_recon)
        new_projs_gt = ct_projector_new.forward_project(gt_volume)
        projs_eval = evaluate_newprojs(new_projs_tr, new_projs_gt)
        CAS_eval['projs'].update(projs_eval)
        # save new_projs to torch pt file
        torch.save(new_projs_tr, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-new_projs_train.pt'))
        torch.save(new_projs_gt, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-new_projs_label.pt'))

        #save model checkpoint
        model_path = os.path.join(config['pretrain_model_path'], str(filename) + "-views-" + str(num_proj) + '-model.pt')
        state_dict = {'net': model.state_dict(), 'enc': encoder.B}
        torch.save(state_dict, model_path)

    # save evaluation results to csv, 并且通过评估结果的平均值的str字符串来命名文件
    # voxel_list = map(lambda x: "{:.3f}".format(x), CAS_eval['voxel'].average())
    # projs_list = map(lambda x: "{:.3f}".format(x), CAS_eval['projs'].average())
    # voxel_str = "-".join([str(number) for number in voxel_list])
    # projs_str = "-".join([str(number) for number in projs_list])
    # CAS_eval['voxel'].save_to_csv(save_dir, f"{algo}-CAS_voxel-views-{num_proj}-{voxel_str}")
    # CAS_eval['projs'].save_to_csv(save_dir, f"{algo}-CAS_projs-views-{num_proj}-{projs_str}")
    ##TODO: 需要修改的地方： filename 是否需要加上str()函数, 数据集名称
    # Test CCTA data
    # best_cldice = 0
    # patient = 0
    # for filename in CCTA_test_list:
    #     print("Start to evaluate " + str(filename) + " with " + str(num_proj) + " views")
    #     best_psnr = 0
    #     patient = 0
    #     # define nerp model
    #     # Setup input encoder:
    #     encoder = Positional_Encoder(config['encoder'])
    #     # Setup model
    #     if config['model'] == 'SIREN':
    #         model = SIREN(config['net'])
    #     elif config['model'] == 'FFN':
    #         model = FFN(config['net'])
    #     else:
    #         raise NotImplementedError
    #     model.cuda()
    #     model.train()
    #
    #     # Load pretrain model
    #     pretrain = False
    #     if pretrain:
    #         model_path = os.path.join(config['pretrain_model_path'], str(filename) + "-views-" + str(num_proj) + '-model.pt')
    #         state_dict = torch.load(model_path)
    #         model.load_state_dict(state_dict['net'])
    #         encoder.B = state_dict['enc']
    #         print('Load pretrain model: {}'.format(model_path))
    #     # Setup optimizer
    #     if config['optimizer'] == 'Adam':
    #         optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']),weight_decay=config['weight_decay'])
    #     else:
    #         NotImplementedError
    #     # Setup loss function
    #     if config['loss'] == 'L2':
    #         loss_fn = torch.nn.MSELoss()
    #     elif config['loss'] == 'L1':
    #         loss_fn = torch.nn.L1Loss()
    #     else:
    #         NotImplementedError
    #     # volume data CAS  & test data  projs
    #     data_path = os.path.join(CCTADataset_path, filename + '_volume.pt')
    #     gt_volume = torch.load(data_path)['volume'][0, :, :, :, :].cuda()  # [1, x, y, z]
    #     input_projs = ct_projector_train.forward_project(gt_volume)  # [1, num_proj, x, y]
    #     # train nerp model
    #     # Input coordinates (x,y) grid and target image
    #     grid = create_grid_3d(*image_size)
    #     grid = grid.cuda()
    #     input_projs = input_projs.cuda()
    #     test_data = (grid, input_projs)  # [bs, z, x, y, 1]
    #     train_data = (grid, input_projs)  # [bs, n, h, w]
    #     scaler = GradScaler()
    #     max_iter = config['max_iter']
    #     for iterations in range(max_iter):
    #         model.train()
    #         optim.zero_grad()
    #         with autocast():
    #             train_embedding = encoder.embedding(train_data[0])
    #             train_output = model(train_embedding)
    #             train_projs = ct_projector_train.forward_project(train_output.transpose(0, 3))  # .squeeze(1))
    #             train_loss = 0.5 * loss_fn(train_projs, train_data[1])
    #         scaler.scale(train_loss).backward()
    #         scaler.step(optim)
    #         scaler.update()
    #         if iterations % 20 == 0:
    #             train_psnr = -10 * torch.log10(2 * train_loss).item()
    #             train_loss = train_loss.item()
    #             print("[Iteration: {}/{}] Train loss: {:.6g} | Train psnr: {:.6g}".format(iterations + 1, max_iter,train_loss,train_psnr))
    #         # 在循环或迭代结束后
    #         del train_embedding, train_output, train_loss
    #         torch.cuda.empty_cache()
    #         # Compute testing psnr
    #         if iterations == 0 or (iterations + 1) % config['val_iter'] == 0:
    #             model.eval()
    #             with torch.no_grad():
    #                 test_embedding = encoder.embedding(test_data[0])
    #                 test_output = model(test_embedding)
    #                 # cldice = cldice_loss(gt_volume, test_output.transpose(0, 3))
    #                 test_projs = ct_projector_train.forward_project(test_output.transpose(0, 3))  # .squeeze(1))
    #                 test_loss = 0.5 * loss_fn(test_projs, test_data[1])
    #                 test_psnr = - 10 * torch.log10(2 * test_loss).item()
    #
    #                 if test_psnr > best_psnr:
    #                     best_psnr = test_psnr
    #                     patient = 0
    #                     saved_model = copy.deepcopy(model.state_dict())
    #                     saved_encoder = copy.deepcopy(encoder.B)
    #                     best_iter = iterations
    #                     print("best_psnr: ", best_psnr)
    #                 else:
    #                     patient += 1
    #                     if patient > 6:
    #                         print("Early stopping at iteration: ", best_iter,"best_psnr: ", best_psnr)
    #                         break
    #
    #     encoder.B = saved_encoder
    #     model.load_state_dict(saved_model)
    #     # save nerp_recon
    #     with torch.no_grad():
    #         test_embedding = encoder.embedding(test_data[0])
    #         train_output = model(test_embedding)
    #     fbp_recon = train_output.transpose(0, 3)
    #     voxel_result = evaluate_voxel(fbp_recon, gt_volume)
    #     CCTA_eval['voxel'].update(voxel_result)
    #     # save fbp_recon
    #     fbp_recon_saved = fbp_recon.squeeze(0).cpu().numpy()
    #     fbp_recon_saved = nib.Nifti1Image(fbp_recon_saved, np.eye(4))
    #     nib.save(fbp_recon_saved, os.path.join(save_dir, filename + "-views-" + str(num_proj) + '-nerp.nii.gz'))
    #
    #     # generate new projs
    #     new_projs_tr = ct_projector_new.forward_project(fbp_recon)
    #     new_projs_gt = ct_projector_new.forward_project(gt_volume)
    #     projs_eval = evaluate_newprojs(new_projs_tr, new_projs_gt)
    #     CCTA_eval['projs'].update(projs_eval)
    #     # save new_projs to torch pt file
    #     torch.save(new_projs_tr, os.path.join(save_dir, filename + "-views-" + str(num_proj) + '-new_projs_train.pt'))
    #     torch.save(new_projs_gt, os.path.join(save_dir, filename + "-views-" + str(num_proj) + '-new_projs_label.pt'))
    #
    #     #save model checkpoint
    #     model_path = os.path.join(config['pretrain_model_path'], str(filename) + "-views-" + str(num_proj) + '-model.pt')
    #     state_dict = {'net': model.state_dict(), 'enc': encoder.B}
    #     torch.save(state_dict, model_path)
    #
    # # save evaluation results to csv, 并且通过评估结果的平均值的str字符串来命名文件
    # voxel_list = map(lambda x: "{:.3f}".format(x), CCTA_eval['voxel'].average())
    # projs_list = map(lambda x: "{:.3f}".format(x), CCTA_eval['projs'].average())
    # voxel_str = "-".join([str(number) for number in voxel_list])
    # projs_str = "-".join([str(number) for number in projs_list])
    # CCTA_eval['voxel'].save_to_csv(save_dir, f"{algo}-CCTA_voxel-views-{num_proj}-{voxel_str}")
    # CCTA_eval['projs'].save_to_csv(save_dir, f"{algo}-CCTA_projs-views-{num_proj}-{projs_str}")

from gaussian_model_anisotropic import GaussianModelAnisotropic
def evaluate_gaussian_fbp(dataset,num_proj, save_dir, opt, args):
    algo = 'gaussian_fbp'
    pretrain_dir = r"/data/xuemingfu/NeRP/gaussian_fbp_pretrain_model"
    # projs_num = [2,4,8,16]
    num_proj = num_proj
    image_size = [128] * 3
    proj_size = [128] * 3
    ct_projector_train = ConeBeam3DProjector(image_size, proj_size, num_proj)
    ct_projector_new = ConeBeam3DProjector(image_size, proj_size, num_proj, start_angle=5)

    eval_res = {'voxel': AverageMeter(),
                'projs': AverageMeter()}
    # Test CAS data
    if dataset == 'CAS':
        test_list = CAS_test_list
        dataset_path = CASDataset_path
    else:
        test_list = CCTA_test_list
        dataset_path = CCTADataset_path
    for filename in test_list:
        print("Start to evaluate " + str(filename) + " with " + str(num_proj) + " views")
        best_psnr = 0
        patient = 0
        best_iter=0
        # prepare gaussian model
        opt.density_lr = args.density_lr
        opt.sigma_lr = args.sigma_lr
        # opt.densify_from_iter= 100
        gaussians = GaussianModelAnisotropic()
        # volume data CAS
        data_path = os.path.join(dataset_path, str(filename) + '_volume.pt')
        gt_volume = torch.load(data_path)['volume'][0,:,:,:,:].cuda()
        input_projs = ct_projector_train.forward_project(gt_volume)   #[1, num_proj, x, y]
        # fbp initial gaussian model
        fbp_recon = ct_projector_train.backward_project(input_projs)
        gaussians.create_from_fbp(fbp_recon, air_threshold=0.05, ini_density=0.04, ini_sigma=0.01, spatial_lr_scale=1, num_samples=args.num_init_gaussian)
        gaussians.training_setup(opt)

        alpha_scheduler = AlphaScheduler(max_iter=args.max_iter - 2000, A=1, B=0)
        mse_loss = torch.nn.functional.mse_loss
        # scaler = GradScaler()
        max_iter = args.max_iter
        starttime = time.time()
        for iteration in range(max_iter):
            # Forward pass
            gaussians.update_learning_rate(iteration)
            # gaussians.optimizer.zero_grad()
            # with autocast():
            # with torch.no_grad():
            grid = create_grid_3d(*image_size)
            grid = grid.cuda()
            # train_data[0] grid: [batchsize, z, x, y, 3]
            grid = grid.unsqueeze(0).repeat(input_projs.shape[0], 1, 1, 1, 1)
            train_output = gaussians.grid_sample(grid, expand=[5, 15, 15])
            #清楚缓存释放gpu
            del grid
            torch.cuda.empty_cache()
            ##loss
            train_projs = ct_projector_train.forward_project(train_output.transpose(1,4).squeeze(1))
            center_mask = extract_vessel_centerline(input_projs, threshold=0)
            center_mask = center_mask > 0

            l1 = mse_loss(train_projs, input_projs)
            l2 = mse_loss(train_projs[center_mask], input_projs[center_mask])
            # 添加l1正则项
            # sparsity_loss = 0.000001 * torch.sum(torch.abs(train_output))
            sparsity_loss = torch.tensor([0]).cuda()
            alpha = alpha_scheduler.log_decay(iteration)
            loss = (1 - alpha) * l1 + alpha * l2 + sparsity_loss
            # scaler.scale(loss).backward()
            # scaler.step(gaussians.optimizer)
            # scaler.update()

            # loss = l1
            loss.backward()
            train_psnr = -10 * torch.log10(l1).item()
            # print(str(filename) + " [Iteration: {}/{}] Train loss: {:.6g} mse loss: {:.6g} centermask_loss: {:.6g} | Train psnr: {:.6g}".format(
            #     iteration + 1, max_iter, loss.item(), l1.item(), l2.item(), train_psnr))

            # Densification
            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 1.5, max_gs_num)
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 1.5)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            #清除缓存释放gpu
            del train_output, train_projs, center_mask, l2, sparsity_loss, alpha, loss
            torch.cuda.empty_cache()

            # if iteration == 0 or (iteration + 1) % 1 == 0:
            if iteration == 0 or (iteration + 1) % 100 == 0:
                # gaussians.eval()
                # with torch.no_grad():
                    # test_psnr = -10 * torch.log10(l1).item()
                if train_psnr > best_psnr:
                    best_psnr = train_psnr
                    patient = 0
                    saved_model = copy.deepcopy(gaussians.state_dict())
                    best_iter = iteration
                    print("best_psnr: ", best_psnr)
                else:
                    patient += 1
                    if patient > 6:
                        print("Early stopping at iteration: ", best_iter, "best_psnr: ", best_psnr)
                        break
        endtime = time.time()
        print("num_views: ", num_proj)
        print("************Training time: s", endtime - starttime)
        gaussians.load_state_dict(saved_model)
        #test
        # gaussians.eval()
        with torch.no_grad():
            grid = create_grid_3d(*image_size)
            grid = grid.cuda()
            # train_data[0] grid: [batchsize, z, x, y, 3]
            grid = grid.unsqueeze(0).repeat(input_projs.shape[0], 1, 1, 1, 1)
            train_output = gaussians.grid_sample(grid, expand=[15, 15, 15])
            #清除缓存释放gpu
            del grid
            torch.cuda.empty_cache()
        # evaluate voxel result
        fbp_recon = train_output.transpose(1,4).squeeze(1).detach()
        voxel_result = evaluate_voxel(fbp_recon, gt_volume)
        eval_res['voxel'].update(voxel_result)
        # save fbp_recon
        fbp_recon_saved = fbp_recon.squeeze(0).detach().cpu().numpy()
        fbp_recon_saved = nib.Nifti1Image(fbp_recon_saved, np.eye(4))
        nib.save(fbp_recon_saved, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '.nii.gz'))
        #generate new projs
        new_projs_tr = ct_projector_new.forward_project(fbp_recon)
        new_projs_gt = ct_projector_new.forward_project(gt_volume)
        projs_eval = evaluate_newprojs(new_projs_tr, new_projs_gt)
        eval_res['projs'].update(projs_eval)
        # save new_projs to torch pt file
        torch.save(new_projs_tr, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-new_projs_train.pt'))
        torch.save(new_projs_gt, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-new_projs_label.pt'))

        #saved model checkpoint
        torch.save(gaussians.state_dict(), os.path.join(pretrain_dir, str(filename) + '-Gaussians_views-' + str(num_proj) + f".pth"))

    # save evaluation results to csv, 并且通过评估结果的平均值的str字符串来命名文件
    # voxel_list = map(lambda x: "{:.3f}".format(x), eval_res['voxel'].average())
    # projs_list = map(lambda x: "{:.3f}".format(x), eval_res['projs'].average())
    # voxel_str = "-".join([str(number) for number in voxel_list])
    # projs_str = "-".join([str(number) for number in projs_list])
    # eval_res['voxel'].save_to_csv(save_dir, f"{algo}-{dataset}_voxel-views-{num_proj}-{voxel_str}")
    # eval_res['projs'].save_to_csv(save_dir, f"{algo}-{dataset}_projs-views-{num_proj}-{projs_str}")

def evaluate_voxelssim(pred_voxel, gt_voxel):
    '''
    pred_voxel: [1, x, y, z]
    gt_voxel: [1, x, y, z]
    '''
    pred_voxel = pred_voxel.squeeze(0)
    gt_voxel = gt_voxel.squeeze(0)

    data_range = gt_voxel.max() - gt_voxel.min()
    # Determine the smaller side of the image
    smaller_side = min(gt_voxel.shape[-2], gt_voxel.shape[-1])
    # Ensure win_size is odd and less than or equal to the smaller side of the image
    win_size = smaller_side if smaller_side % 2 == 1 else smaller_side - 1
    # Calculate SSIM
    ssim_val = ssim(gt_voxel.cpu().numpy(), pred_voxel.cpu().numpy(), multichannel=True,data_range=data_range.cpu().item(), win_size=win_size)

    return ssim_val
import hydra
from omegaconf import DictConfig
from gaussian_predictor import GaussianSplatPredictor
from general_utils import safe_state
@hydra.main(version_base=None, config_path='configs', config_name="default_config_oral")
def main(cfg: DictConfig):
    algo = 'gaussian_splatter'
    save_dir = cfg.comparison.save_dir
    pretrain_dir = cfg.comparison.pretrain_model_path
    projs_num = [16]
    for i in projs_num:
        # num_proj = cfg.comparison.num_proj
        num_proj = i
        image_size = [128] * 3
        proj_size = [128] * 3
        ct_projector_train = ConeBeam3DProjector(image_size, proj_size, num_proj)
        ct_projector_new = ConeBeam3DProjector(image_size, proj_size, num_proj, start_angle=5)

        eval_res = {'voxel': AverageMeter(),
                    'projs': AverageMeter()}

        device = safe_state(cfg)
        #定义模型
        gaussian_predictor = GaussianSplatPredictor(cfg)
        gaussian_predictor.to(device)

        #加载预训练模型
        # pretrained_ckpt_dir = os.path.join('./', "best_modelv5.pth")
        pretrained_ckpt_dir = os.path.join(pretrain_dir, cfg.comparison.gaussian_predictor_pretrained_model)
        checkpoint = torch.load(pretrained_ckpt_dir, map_location=device)
        gaussian_predictor.load_state_dict(checkpoint["model_state_dict"], strict=True)

        # Test CAS data
        if cfg.comparison.dataset == 'CAS':
            test_list = CAS_test_list
            dataset_path = CASDataset_path
        else:
            test_list = CCTA_test_list
            dataset_path = CCTADataset_path

        for filename in test_list:
            print("Start to evaluate " + str(filename) + " with " + str(num_proj) + " views")
            best_ssim = -100
            patient = 0
            best_iter = 0
            # volume data CASs
            data_path = os.path.join(dataset_path, str(filename) + '_volume.pt')
            data_path1 = os.path.join(dataset_path, str(filename) + '.pt')
            gt_volume = torch.load(data_path)['volume'][0,:,:,:,:].cuda()
            points_gt = torch.load(data_path1)['bg_mask'][0].cuda()
            depth_gt = torch.load(data_path1)['depth'][0].cuda()
            # save_depth_image(depth_gt, cmap='viridis', save_path="/media/I/xcw/3DGR-CAR-main/3dgs-car/our_point_vs_gt_point/depth_map.png")
            input_projs = ct_projector_train.forward_project(gt_volume)   #[1, num_proj, x, y]

            # gaussian predictor initial
            out_dict, depth = gaussian_predictor(input_projs[:,0,:,:].reshape((1,1,1,128,128)))
            _, depth2 = gaussian_predictor(input_projs[:, 0, :, :].reshape((1, 1, 1, 128, 128)))

            points = out_dict['_xyz'].detach()
            num_samples = 2048
            indices = torch.randperm(points.size(0))[:num_samples]
            points = points[indices, ...]

            gaussians = GaussianModelAnisotropic()
            gaussians.create_from_points_cloud(points, spatial_lr_scale=1)
            # ###初始化高斯点云可视化
            # points = gaussians._xyz  # 或 gaussians.get_xyz()
            # visualize_pointcloud(points, save_path="initial_pointcloud.png")
            ###高斯GIF
            # create_rotating_pointcloud_gif(
            #                                 xyz=gaussians._xyz,
            #                                 density=gaussians._density,
            #                                 scaling=gaussians._scaling,
            #                                 save_path="initial_gaussians_dsg_1.gif",
            #                                 n_frames=36
            #                             )
            # create_rotating_pointcloud_gif(
            #                                 xyz=points_gt,
            #                                 density=None,
            #                                 scaling=None,
            #                                 save_path="initial_gaussians_gt.gif",
            #                                 n_frames=36
            #                             )
            ###############点云可视化代码
            # PointCloudVisualizer1.save_static_pointcloud(
            #                                                 gaussians._xyz,
            #                                                 density=gaussians._density,
            #                                                 scaling=gaussians._scaling,
            #                                                 save_path="/media/I/xcw/3DGR-CAR-main/3dgs-car/baseline_point_vs_gt_point/baseline_asoca.png",
            #                                                 point_size=1,
            #                                                 view_elev=0,
            #                                                 view_azim=-270   # 向左45度
            #                                             )

            # PointCloudVisualizer2.save_static_pointcloud(
            #                                                 points_gt,
            #                                                 density=None,
            #                                                 scaling=None,
            #                                                 save_path="/media/I/xcw/3DGR-CAR-main/3dgs-car/baseline_point_vs_gt_point/gt_asoca.png",
            #                                                 point_size=5,
            #                                                 view_elev=0,
            #                                                 view_azim=-90   # 向左45度
            #                                             )
            ###############点云可视化代码
            # PointCloudVisualizer.save_pointclouds_together(
            #                                                 "/media/I/xcw/3DGR-CAR-main/3dgs-car/our_point_vs_gt_point",
            #                                                 gaussians._xyz, points_gt,
            #                                                 filename=filename
            #                                             )
            gaussians.training_setup(cfg.Gaussians_Opt)

            # alpha_scheduler = AlphaScheduler(max_iter=10000, A=1, B=0)
            alpha_scheduler = AlphaScheduler(max_iter=6000, A=1, B=0)
            # lpips_scheduler = AlphaScheduler(max_iter=2000, A=0.1, B=1)
            mse_loss = torch.nn.functional.mse_loss

            first_iter = 0
            max_iter = cfg.comparison.max_iter
            starttime = time.time()
            for iteration in range(first_iter, max_iter):
                # Forward pass
                gaussians.update_learning_rate(iteration)

                grid = create_grid_3d(*image_size)
                grid = grid.cuda()
                # train_data[0] grid: [batchsize, z, x, y, 3]
                grid = grid.unsqueeze(0).repeat(input_projs.shape[0], 1, 1, 1, 1)
                # grid = grid.unsqueeze(-2)
                train_output = gaussians.grid_sample(grid, expand=[5, 15, 15])
                del grid
                torch.cuda.empty_cache()
                ##loss
                train_projs = ct_projector_train.forward_project(train_output.transpose(1, 4).squeeze(1))
                center_mask = extract_vessel_centerline(input_projs, threshold=0)
                center_mask = center_mask > 0

                depth_pred = voxel_to_depth_map_differentiable(train_output.transpose(1, 4).squeeze(1))
                depth_pred2 = voxel_to_depth_map_differentiable_2(train_output.transpose(1, 4).squeeze(1))

                l1 = mse_loss(train_projs, input_projs)
                l2 = mse_loss(train_projs[center_mask], input_projs[center_mask])
                # l3 = mse_loss(depth_pred[::2,::2], depth[0, 0, :, :]) + mse_loss(depth_pred2[::2,::2], depth2[0, 0, :, :])
                # l3 = mse_loss(depth_pred[::2,::2], depth[0, 0, :, :])
                l3 = torch.tensor([0]).cuda()
                # 添加l1正则项
                # sparsity_loss = 0.000001 * torch.sum(torch.abs(train_output))

                sparsity_loss = torch.tensor([0]).cuda()

                alpha = alpha_scheduler.log_decay(iteration)
                # alpha_lpips = lpips_scheduler.iterative_decay(iteration)
                # loss = 0.5*l1 + 0.5*l2 + sparsity_loss + alpha_lpips*l3
                loss = (1 - alpha) * l1 + alpha * l2 + sparsity_loss + l3
                # loss = l1
                loss.backward(retain_graph=True)
                train_psnr = -10 * torch.log10(l1.detach()).item()
                # with torch.no_grad():
                # Densification
                if iteration < cfg.Gaussians_Opt.densify_until_iter:
                    if iteration > 500 and iteration % cfg.Gaussians_Opt.densification_interval == 0:
                        # gaussians.only_prune(cfg.Gaussians_Opt.densify_grad_threshold, 0.005, 1.5)
                        gaussians.densify_and_prune(cfg.Gaussians_Opt.densify_grad_threshold, 0.005, 1.5)
                        # 统计高斯模型中一共有多少个gaussians
                        # print("Number of gaussians after densification: ", gaussians.get_gaussians_num)

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                #清除缓存释放gpu
                del train_projs,
                torch.cuda.empty_cache()
                if iteration == 0 or (iteration + 1) % 100 == 0:
                    #print train_psnr
                    ssim = evaluate_voxelssim(train_output.transpose(1, 4).squeeze(0).detach(), gt_volume.detach())
                    print(str(filename) + " [Iteration: {}/{}] Train loss: {:.6g} mse loss: {:.6g} centermask_loss: {:.6g} sparsity_loss: {:.6g} l3_loss: {:.6g}| Train psnr: {:.6g} | SSIM: {:.6g}".format(
                            iteration + 1, cfg.Gaussians_Opt.iterations, loss.item(), l1.item(), l2.item(),sparsity_loss.item(), l3.item(), train_psnr, ssim))
                    # print("num_views: ", num_proj)
                    endtime = time.time()
                    print("************to now  Training time: s", endtime - starttime)
                    if ssim > best_ssim:
                        best_ssim = ssim
                        patient = 0
                        saved_model = copy.deepcopy(gaussians.state_dict())
                        best_iter = iteration
                        print("best_ssim: ", best_ssim)

                    else:
                        patient += 1
                        if patient > 10:
                            print("Early stopping at iteration: ", best_iter, "best_ssim: ", best_ssim)
                            break
                    # if train_psnr > 22:
                    #     break


            endtime = time.time()
            print("num_views: ", num_proj)
            print("************Training time: s", endtime - starttime)
            gaussians.load_state_dict(saved_model)
            #test
            # gaussians.eval()
            with torch.no_grad():
                grid = create_grid_3d(*image_size)
                grid = grid.cuda()
                # train_data[0] grid: [batchsize, z, x, y, 3]
                grid = grid.unsqueeze(0).repeat(input_projs.shape[0], 1, 1, 1, 1)
                train_output = gaussians.grid_sample(grid, expand=[15, 15, 15])
                #清除缓存释放gpu
                del grid
                torch.cuda.empty_cache()
            # evaluate voxel result
            fbp_recon = train_output.transpose(1, 4).squeeze(1).detach()
            voxel_result = evaluate_voxel(fbp_recon, gt_volume)
            eval_res['voxel'].update(voxel_result)
            # save fbp_recon
            fbp_recon_saved = fbp_recon.squeeze(0).detach().cpu().numpy()
            fbp_recon_saved = nib.Nifti1Image(fbp_recon_saved, np.eye(4))
            nib.save(fbp_recon_saved, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '.nii.gz'))
            # generate new projs
            new_projs_tr = ct_projector_new.forward_project(fbp_recon)
            new_projs_gt = ct_projector_new.forward_project(gt_volume)
            projs_eval = evaluate_newprojs(new_projs_tr, new_projs_gt)
            eval_res['projs'].update(projs_eval)
            # save new_projs to torch pt file
            torch.save(new_projs_tr, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-new_projs_train.pt'))
            torch.save(new_projs_gt,os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-new_projs_label.pt'))

            # saved model checkpoint
            torch.save(gaussians.state_dict(),os.path.join(pretrain_dir, str(filename) + '-Gaussians_views-' + str(num_proj) + f"{train_psnr:.2f}.pth"))

    # save evaluation results to csv, 并且通过评估结果的平均值的str字符串来命名文件
    voxel_list = map(lambda x: "{:.3f}".format(x), eval_res['voxel'].average())
    projs_list = map(lambda x: "{:.3f}".format(x), eval_res['projs'].average())
    voxel_str = "-".join([str(number) for number in voxel_list])
    projs_str = "-".join([str(number) for number in projs_list])
    eval_res['voxel'].save_to_csv(save_dir, f"{algo}-{cfg.comparison.dataset}_voxel-views-{num_proj}-{voxel_str}")
    eval_res['projs'].save_to_csv(save_dir, f"{algo}-{cfg.comparison.dataset}_projs-views-{num_proj}-{projs_str}")



if __name__ == "__main__":
    # yaml_path = './configs/default_config.yaml'
    import time
    cpu_num = 10
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    # 加载数据  newdata: 最新的数据集(更新了降采样方法)    new: 之前的数据集
    # CCTADataset_path = r"/data/xuemingfu/project1/Coronary_Angiography_data/torchdata_newdata"
    CCTADataset_path = "/media/I/xcw/3DGR-CAR-main/all_data/CCTA/torchdata_newdata"

    with open("/media/I/xcw/3DGR-CAR-main/3dgs-car/data_json/CCTAData.json", 'r') as load_f:
        load_dict = json.load(load_f)
        # train_list = load_dict['train']
        # CCTA_val_list = load_dict['validation']
        CCTA_test_list = load_dict['test']
    # CASDataset_path = r'/data/xuemingfu/project1/Coronary_Angiography_data/ImageCAS/torchdata_newdata'
    CASDataset_path = "/media/I/xcw/3DGR-CAR-main/all_data/ImageCAS/torchdata_newdata"
    with open("/media/I/xcw/3DGR-CAR-main/3dgs-car/data_json/imageCAS.json", 'r') as load_f:
        load_dict = json.load(load_f)
        # CAS_train_list = load_dict['train']
        # CAS_val_list = load_dict['validation']
        CAS_test_list = load_dict['test']
        # CAS_test_list = load_dict['validation']
    max_gs_num = 10000
    CCTADataset_path = "/media/I/xcw/3DGR-CAR-main/3dgs-car"
    CCTA_test_list = ['Normal_1.mha']
    # #TODO: fbp
    # for num_proj in [2,4,8, 16]:
    #     fbp_dir = r"/data/xuemingfu/NeRP/fbp_result"
    #     evaluate_fbp(num_proj = num_proj, save_dir=fbp_dir)
    # ##TODO: nerp
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='./configs/ct_recon_3d.yaml', help='Path to the config file.')
    # # Load experiment setting for nerp
    # nerp_opts = parser.parse_args()
    # nerp_dir = r"/data/xuemingfu/NeRP/nerp_result"
    # for i in [2,4,8,16]:
    #     evaluate_nerp(num_proj = i, save_dir=nerp_dir, config_file=nerp_opts.config)
    import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='./configs/ct_recon_3d.yaml', help='Path to the config file.')
    # # Load experiment setting for nerp
    # nerp_opts = parser.parse_args()
    # nerp_dir = r"/data/xuemingfu/NeRP/nerp_result"
    # for i in [2,16]:
    #     evaluate_nerp(num_proj = i, save_dir=nerp_dir, config_file=nerp_opts.config)
    ##TODO: gaussian_fbp
    import sys
    # from arguments_init import *
    # parser = ArgumentParser(description="Training script parameters")
    # lp = ModelParams(parser)
    # op = OptimizationParams(parser)
    # pp = PipelineParams(parser)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    # parser.add_argument("--start_checkpoint", type=str, default = None)
    #
    # parser.add_argument('--mydensity_lr', type=float, default=1e-2)
    # parser.add_argument('--mysigma_lr', type=float, default=1e-2)
    #
    # parser.add_argument('--max_iter', type=int, default=8000)
    # parser.add_argument('--num_init_gaussian', type=int, default=10000)
    # parser.add_argument('--num_proj', type=int, default=16)
    # args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    #
    # gaussian_fbp_dir = r"/data/xuemingfu/NeRP/gaussian_fbp_result"
    # # dataset = 'CCTA' # 'CAS'
    # # evaluate_gaussian_fbp(dataset, args.num_proj, gaussian_fbp_dir, op.extract(args), args)
    # for num_proj in [2,16]:
    #     args.num_proj = num_proj
    #     dataset = 'CAS'
    #     evaluate_gaussian_fbp(dataset, args.num_proj, gaussian_fbp_dir, op.extract(args), args)

    #TODO: gaussian_splatter
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_proj', type=int, default=2)
    parser.add_argument('--datasetname', type=str, default='CAS')
    # 修改 default_config.ymal 中的 comparison参数
    main()




