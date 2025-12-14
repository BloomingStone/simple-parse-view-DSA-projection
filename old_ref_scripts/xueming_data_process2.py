import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import os

import SimpleITK as sitk
import torch.nn.functional as F
import numpy as np
from ct_geometry_projector import ConeBeam3DProjector
from scipy.ndimage import zoom,gaussian_filter
from skimage import morphology
from torch.utils.data import Dataset
import nibabel as nib
import gc
from cldice import soft_skel, soft_dice, soft_cldice
from skimage.morphology import skeletonize_3d

def sample_closest_points(points, num_samples):
    # 随机选择点的索引
    indices = torch.randperm(points.size(0))[:num_samples]
    # 根据索引选择点
    sampled_points = points[indices]
    return sampled_points

def points_to_volume(points, volume_size=128):

    points = (points * (volume_size - 1)).long()

    # 创建一个全零的体素数据
    volume = torch.zeros(1, volume_size, volume_size, volume_size, device=points.device)

    volume[0, points[:, 0], points[:, 1], points[:, 2]] = 1
    return volume

def chamfer_distance(point_cloud1, point_cloud2):

    dist1 = torch.cdist(point_cloud1, point_cloud2)
    dist2 = torch.cdist(point_cloud2, point_cloud1)

    # 计算到最近点的距离的平方
    chamfer_dist1 = torch.mean(torch.min(dist1, dim=1).values)
    chamfer_dist2 = torch.mean(torch.min(dist2, dim=1).values)

    # 计算Chamfer Distance
    chamfer_distance = chamfer_dist1 + chamfer_dist2

    return chamfer_distance

def centerline_loss(pred_voxel, target_voxel):
    # 计算骨架
    pred_skel = soft_skel(pred_voxel, 3)
    target_skel = soft_skel(target_voxel, 3)
    # 计算基于骨架的损失，例如Dice损失
    return soft_dice(pred_skel, target_skel)

def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid
def resample_to_standard(scan, scale):
    # 计算原始尺寸和目标尺寸之间的缩放因子
    z, y, x = scan.shape
    z_scale, y_scale, x_scale = scale / z, scale / y, scale / x

    # 使用scipy的zoom函数进行重采样
    # order=1 表示使用线性插值，这在很多情况下是一个好的折中选择
    # mode='nearest' 表示边界外的值将通过最近的边界值进行估计
    resampled_scan = zoom(scan, (z_scale, y_scale, x_scale), order=3, mode='nearest')
    return resampled_scan
def resample_valid_voxels_to_standard(scan, scale):
    # 计算原始尺寸和目标尺寸之间的缩放因子
    z, y, x = scan.shape
    z_scale, y_scale, x_scale = scale / z, scale / y, scale / x

    # 创建一个掩码以标记背景位置
    background_mask = scan == 0

    # 重采样整个数据集
    resampled_scan = zoom(scan, (z_scale, y_scale, x_scale), order=1, mode='nearest')

    # 对掩码也进行相同的重采样，并应用于重采样后的数据
    resampled_background_mask = zoom(background_mask.astype(int), (z_scale, y_scale, x_scale), order=0, mode='nearest').astype(bool)
    resampled_scan[resampled_background_mask] = 0

    return resampled_scan

def crop_to_standard(scan, scale):
    z, y, x = scan.shape
    if z >= scale:
        ret_scan = scan[z-scale:z, :, :]
        # ret_scan = scan
    else:
        temp1 = np.zeros(((scale-z)//2, y, x))
        temp2 = np.zeros(((scale-z)-(scale-z)//2, y, x))
        ret_scan = np.concatenate((temp1, scan, temp2), axis=0)
    z, y, x = ret_scan.shape
    if y >= scale:
        ret_scan = ret_scan[:, (y-scale)//2:(y+scale)//2, :]
    else:
        temp1 = np.zeros((z, (scale-y)//2, x))
        temp2 = np.zeros((z, (scale-y)-(scale-y)//2, x))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=1)
    z, y, x = ret_scan.shape
    if x >= scale:
        ret_scan = ret_scan[:, :, (x-scale)//2:(x+scale)//2]
    else:
        temp1 = np.zeros((z, y, (scale-x)//2))
        temp2 = np.zeros((z, y, (scale-x)-(scale-x)//2))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=2)
    return ret_scan
def recenter_voxels(scan):
    # Step 1: Find the bounds of the non-zero voxels
    x_coords, y_coords, z_coords = np.nonzero(scan)
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()
    min_z, max_z = z_coords.min(), z_coords.max()

    # Step 2: Extract the relevant cube of data
    cropped = scan[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

    # Step 3: Calculate the center of the cropped data and recenter it in the original scan
    center_x, center_y, center_z = (max_x + min_x) // 2, (max_y + min_y) // 2, (max_z + min_z) // 2
    original_center = np.array(scan.shape) // 2
    cropped_center = np.array(cropped.shape) // 2

    # Create an empty array of the same shape as the original scan
    recentered_scan = np.zeros_like(scan)

    # Calculate where to place the cropped data in the original scan
    start_x = original_center[0] - cropped_center[0]
    start_y = original_center[1] - cropped_center[1]
    start_z = original_center[2] - cropped_center[2]

    end_x = start_x + cropped.shape[0]
    end_y = start_y + cropped.shape[1]
    end_z = start_z + cropped.shape[2]

    # Place the cropped data into the recentered_scan
    recentered_scan[start_x:end_x, start_y:end_y, start_z:end_z] = cropped

    return recentered_scan
def crop_to_standard_cas(source_scan, scale):

    scan = recenter_voxels(source_scan)
    x, y, z = scan.shape
    # 处理Z轴
    if z >= scale:
        start_z = (z - scale) // 2
        ret_scan = scan[:, :, start_z:start_z + scale]
    else:
        temp1 = np.zeros((x, y, (scale - z) // 2))
        temp2 = np.zeros((x, y, (scale - z) - (scale - z) // 2))
        ret_scan = np.concatenate((temp1, scan, temp2), axis=2)

    x, y, z = ret_scan.shape

    # 处理Y轴
    if y >= scale:
        start_y = (y - scale) // 2
        ret_scan = ret_scan[:, start_y:start_y + scale, :]
    else:
        temp1 = np.zeros((x, (scale - y) // 2, z))
        temp2 = np.zeros((x, (scale - y) - (scale - y) // 2, z))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=1)

    x, y, z = ret_scan.shape

    # 处理X轴
    if x >= scale:
        start_x = (x - scale) // 2
        ret_scan = ret_scan[start_x:start_x + scale, :, :]
    else:
        temp1 = np.zeros(((scale - x) // 2, y, z))
        temp2 = np.zeros(((scale - x) - (scale - x) // 2, y, z))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=0)

    return ret_scan
#Old version
# def crop_to_standard_cas(source_scan, scale):
#
#     scan = recenter_voxels(source_scan)
#     x, y, z = scan.shape
#     # 处理Z轴
#     if z >= scale:
#         ret_scan = scan[:, :, z - scale:z]
#     else:
#         temp1 = np.zeros((x, y, (scale - z) // 2))
#         temp2 = np.zeros((x, y, (scale - z) - (scale - z) // 2))
#         ret_scan = np.concatenate((temp1, scan, temp2), axis=2)
#
#     x, y, z = ret_scan.shape
#
#     # 处理Y轴
#     if y >= scale:
#         ret_scan = ret_scan[:, y - scale:y, :]
#     else:
#         temp1 = np.zeros((x, (scale - y) // 2, z))
#         temp2 = np.zeros((x, (scale - y) - (scale - y) // 2, z))
#         ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=1)
#
#     x, y, z = ret_scan.shape
#
#     # 处理X轴
#     if x >= scale:
#         ret_scan = ret_scan[x - scale:x, :, :]
#     else:
#         temp1 = np.zeros(((scale - x) // 2, y, z))
#         temp2 = np.zeros(((scale - x) - (scale - x) // 2, y, z))
#         ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=0)
#
#     return ret_scan

def crop_to_standard_liver(scan, scale):

    x, y, z = scan.shape
    # 处理Z轴
    if z >= scale:
        ret_scan = scan[:, :, z - scale:z]
    else:
        temp1 = np.zeros((x, y, (scale - z) // 2))
        temp2 = np.zeros((x, y, (scale - z) - (scale - z) // 2))
        ret_scan = np.concatenate((temp1, scan, temp2), axis=2)

    x, y, z = ret_scan.shape

    # 处理Y轴
    if y >= scale:
        ret_scan = ret_scan[:, y - scale:y, :]
    else:
        temp1 = np.zeros((x, (scale - y) // 2, z))
        temp2 = np.zeros((x, (scale - y) - (scale - y) // 2, z))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=1)

    x, y, z = ret_scan.shape

    # 处理X轴
    if x >= scale:
        ret_scan = ret_scan[x - scale:x, :, :]
    else:
        temp1 = np.zeros(((scale - x) // 2, y, z))
        temp2 = np.zeros(((scale - x) - (scale - x) // 2, y, z))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=0)

    return ret_scan

#
def extract_vessel_centerline(train_projs, threshold):
    """
    Extract the centerline of vessels in each 2D projection in train_projs.

    Args:
        train_projs (torch.Tensor): Tensor of shape (B, num_proj, proj_size_h, proj_size_w)
        threshold (float): Threshold for binarization

    Returns:
        torch.Tensor: Tensor of the same shape as train_projs, with 1s at the centerline of vessels and 0s elsewhere.
    """
    # Convert to numpy and move to CPU for skimage functions
    train_projs_np = train_projs.cpu().numpy()

    # Initialize an empty array for the centerline masks
    centerline_masks = np.zeros_like(train_projs_np)

    # Iterate over each projection
    for i in range(train_projs_np.shape[0]):
        for j in range(train_projs_np.shape[1]):
            # Binarize the projection using the provided threshold
            binary_proj = train_projs_np[i, j] > threshold

            # Extract the centerline (skeletonize)
            centerline = morphology.skeletonize(binary_proj)

            # Store the centerline mask
            centerline_masks[i, j] = centerline

    # Convert the centerline masks back to a PyTorch tensor
    centerline_masks = torch.from_numpy(centerline_masks).to(train_projs.device)

    return centerline_masks
# define the dataset class
class CCTADataset_generate(Dataset):
    def __init__(self, dir_path, filenames ,scale_size=128, num_proj=16):
        super().__init__()
        self.dir_path = dir_path
        #file_list = os.listdir(self.dir_path)
        #for file in file_list:
        #    if not file.endswith(".mha"):
            #选择以Normal 开头的文件
            # if not file.startswith("Normal"):
        #        file_list.remove(file)
        #self.file_list = file_list
        self.file_list = filenames
        self.scale_size = scale_size
        self.num_proj = num_proj
        self.image_size = [self.scale_size]*3
        self.proj_size = [self.scale_size]*3
        self.ct_projector = ConeBeam3DProjector(self.image_size, self.proj_size, self.num_proj)
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        mha_file = os.path.join(self.dir_path, self.file_list[index])
        itkimage = sitk.ReadImage(mha_file)
        image_mha = sitk.GetArrayFromImage(itkimage)
        image_mha = crop_to_standard_cas(image_mha, scale=320)
        image_mha = resample_valid_voxels_to_standard(image_mha, self.scale_size)
        mask = image_mha == 0
        image_mha = torch.tensor(image_mha, dtype=torch.float32)[None, ...]
        mask = torch.tensor(mask, dtype=torch.float32)[None, ...]
        image_mha = F.interpolate(image_mha, size=(self.scale_size, self.scale_size), mode='bilinear', align_corners=False)
        resampled_mask  = F.interpolate(mask, size=(self.scale_size, self.scale_size), mode='bilinear', align_corners=False)
        image_mha[resampled_mask.bool()] = 0
        image_mha = image_mha / torch.max(image_mha)  # [B, C, H, W], [0, 1]
        # mean = image_mha.mean()
        # std = image_mha.std()
        # image_mha = (image_mha - mean) / std

        # fbp_recon_g = image_mha[0, :, :, :].numpy()
        # fbp_recon_g = nib.Nifti1Image(fbp_recon_g, np.eye(4))
        # nib.save(fbp_recon_g, self.file_list[index] + "testtest.nii.gz")

        image_mha = image_mha.permute(1, 2, 3, 0)  # [C, H, W, 1]
        image_mha = image_mha.unsqueeze(0)
        image_mha = image_mha.cuda()
        projs = self.ct_projector.forward_project(image_mha.transpose(1, 4).squeeze(1))
        # x = torch.tensor(projs[0, 0, :, :]).reshape( 1, 1, 128, 128).cuda()
        x = projs[0, 0, :, :].clone().detach().reshape(1, 1, self.scale_size, self.scale_size).cuda()
        return x, projs.squeeze(0)

class CASDataset_generate(Dataset):
    def __init__(self, dir_path=r'/data/xuemingfu/project1/Coronary_Angiography_data/ImageCAS/dataset', filenames=None ,scale_size=128, num_proj=16):
        super().__init__()
        self.dir_path = dir_path
        #file_list = os.listdir(self.dir_path)
        #for file in file_list:
        #    if not file.endswith(".mha"):
            #选择以Normal 开头的文件
            # if not file.startswith("Normal"):
        #        file_list.remove(file)
        #self.file_list = file_list
        self.file_list = filenames
        self.scale_size = scale_size
        self.num_proj = num_proj
        self.image_size = [self.scale_size]*3
        self.proj_size = [self.scale_size]*3
        self.ct_projector = ConeBeam3DProjector(self.image_size, self.proj_size, self.num_proj)
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = os.path.join(self.dir_path, str(self.file_list[index]), 'img.nii.gz')
        label_path = os.path.join(self.dir_path, str(self.file_list[index]), 'label.nii.gz')
        image = nib.load(file_path)
        labels = nib.load(label_path)
        image_mha = image.get_fdata()*(labels.get_fdata()==1)
        image_mha[image_mha < 0] = 0
        image_mha = crop_to_standard_cas(image_mha, scale=512)
        image_mha = resample_valid_voxels_to_standard(image_mha, self.scale_size)
        mask = image_mha == 0
        image_mha = torch.tensor(image_mha, dtype=torch.float32)[None, ...]
        mask = torch.tensor(mask, dtype=torch.float32)[None, ...]
        image_mha = F.interpolate(image_mha, size=(self.scale_size, self.scale_size), mode='bilinear', align_corners=False)
        resampled_mask  = F.interpolate(mask, size=(self.scale_size, self.scale_size), mode='bilinear', align_corners=False)
        image_mha[resampled_mask.bool()] = 0
        image_mha = image_mha / torch.max(image_mha)  # [B, C, H, W], [0, 1]
        # mean = image_mha.mean()
        # std = image_mha.std()
        #
        # image_mha = (image_mha - mean) / std

        image_mha = image_mha.permute(1, 2, 3, 0)  # [C, H, W, 1]
        image_mha = image_mha.unsqueeze(0)
        image_mha = image_mha.cuda()

        projs = self.ct_projector.forward_project(image_mha.transpose(1, 4).squeeze(1))
        # x = torch.tensor(projs[0, 0, :, :]).reshape( 1, 1, 128, 128).cuda()
        x = projs[0, 0, :, :].clone().detach().reshape(1, 1, self.scale_size, self.scale_size).cuda()
        return x, projs.squeeze(0)

from torch.utils.data import DataLoader as Dataloader
class ForeverDataIterator:
    def __init__(self, data_loader: Dataloader, device=None):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            data = next(self.data_iter)
        return data
    def __len__(self):
        return len(self.data_loader)

# define the dataset class
def rotation_matrix_x(theta):
    theta = torch.tensor(theta, dtype=torch.float32) * torch.pi / 180.0
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return torch.tensor([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ], dtype=torch.float32)

def rotation_matrix_y(theta):
    theta = torch.tensor(theta, dtype=torch.float32) * torch.pi / 180.0
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return torch.tensor([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ], dtype=torch.float32)
def rotation_matrix_z(theta):
    # 将角度从度转换为弧度，并确保是Tensor类型
    theta = torch.tensor(theta, dtype=torch.float32) * torch.pi / 180.0
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return torch.tensor([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=torch.float32)

def rotate_point(points, theta, axis='z'):
    if axis == 'x':
        R = rotation_matrix_x(theta)
    elif axis == 'y':
        R = rotation_matrix_y(theta)
    elif axis == 'z':
        R = rotation_matrix_z(theta)
    return torch.mm(points, R.t())
# 旋转点云来模拟坐标系的旋转
def rotate_coordinate_system(points, theta):
    R = rotation_matrix_z(theta)
    return torch.mm(points, R.t())
# class CCTADataset(Dataset):
#     def __init__(self, dir_path, filenames):
#         super().__init__()
#         self.dir_path = dir_path
#         self.file_list = filenames
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, index):
#         data = torch.load(os.path.join(self.dir_path, self.file_list[index][:-4] + '.pt'))
#         points = np.argwhere(1- data['bg_mask'].squeeze(0)).T/128
#         return data['x'], data['projs'], points
#
# class CASDataset(Dataset):
#     def __init__(self, dir_path=r'/data/xuemingfu/project1/Coronary_Angiography_data/ImageCAS/torchdata', filenames=None ):
#         super().__init__()
#         self.dir_path = dir_path
#         self.file_list = filenames
#     def __len__(self):
#         return len(self.file_list)*16
#
#     def __getitem__(self, index):
#         data = torch.load(os.path.join(self.dir_path, str(self.file_list[index//16]) + '.pt'))
#         points = np.argwhere(1- data['bg_mask'].squeeze(0)).T/128
#
#         return data['projs'][index%16, :, :].reshape(1, 1, 128, 128), data['projs'], rotate_coordinate_system(points, 11.25*(index%16))
#

class CCTADataset(Dataset):
    def __init__(self, dir_path, filenames):
        super().__init__()
        self.dir_path = dir_path
        self.file_list = filenames
        self.data_cache = []
        self.load_data_into_memory()

    def load_data_into_memory(self):
        for filename in self.file_list:
            data_path = os.path.join(self.dir_path, filename[:-4] + '.pt')
            data = torch.load(data_path)
            points = np.argwhere(1 - data['bg_mask'].squeeze(0)).T / 128
            self.data_cache.append((data['x'], data['projs'], points))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.data_cache[index]
class CASDataset(Dataset):
    def __init__(self, dir_path=r'/data/xuemingfu/project1/Coronary_Angiography_data/ImageCAS/torchdata', filenames=None):
        super().__init__()
        self.dir_path = dir_path
        self.file_list = filenames
        self.data_cache = []
        self.load_data_into_memory()

    def load_data_into_memory(self):
        for filename in self.file_list:
            data_path = os.path.join(self.dir_path, str(filename) + '.pt')
            data = torch.load(data_path)
            points = np.argwhere(1 - data['bg_mask'].squeeze(0)).T / 128
            self.data_cache.append((data['x'], data['projs'], points))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.data_cache[index]

# class CCTADataset(Dataset):
#     def __init__(self, dir_path, filenames):
#         super().__init__()
#         self.dir_path = dir_path
#         self.file_list = filenames
#         self.data_cache = []
#         self.load_data_into_memory()
#
#     def load_data_into_memory(self):
#         for filename in self.file_list:
#             data_path = os.path.join(self.dir_path, filename[:-4] + '.pt')
#             data = torch.load(data_path)
#             points = np.argwhere(1 - data['bg_mask'].squeeze(0)).T / 128
#             for i in range(16):  # Assuming each file corresponds to 16 data points
#                 rotated_points = rotate_coordinate_system(points, 11.25 * i)
#                 self.data_cache.append((data['projs'][i, :, :].reshape(1, 1, 128, 128), data['projs'], rotated_points))
#
#     def __len__(self):
#         return len(self.file_list) * 16  # 16 data points per file
#
#     def __getitem__(self, index):
#         return self.data_cache[index]
# class CASDataset(Dataset):
#     def __init__(self, dir_path=r'/data/xuemingfu/project1/Coronary_Angiography_data/ImageCAS/torchdata', filenames=None):
#         super().__init__()
#         self.dir_path = dir_path
#         self.file_list = filenames
#         self.data_cache = []
#         self.load_data_into_memory()
#
#     def load_data_into_memory(self):
#         for filename in self.file_list:
#             data_path = os.path.join(self.dir_path, str(filename) + '.pt')
#             data = torch.load(data_path)
#             points = np.argwhere(1 - data['bg_mask'].squeeze(0)).T / 128
#             for i in range(16):
#                 rotated_points = rotate_coordinate_system(points, 11.25 * i)
#                 self.data_cache.append((data['projs'][i, :, :].reshape(1, 1, 128, 128), data['projs'], rotated_points))
#
#     def __len__(self):
#         return len(self.file_list) * 16
#
#     def __getitem__(self, index):
#         return self.data_cache[index]


class AlphaScheduler:
    def __init__(self, max_iter, A, B):
        self.max_iter = max_iter
        self.A = A
        self.B = B

    def linear_decay(self, current_iter):
        alpha = self.A + (self.B - self.A) * current_iter / self.max_iter
        return max(min(alpha, self.A), self.B)

    def log_decay(self, current_iter):
        alpha = self.A + (self.B - self.A) * np.log(current_iter + 1) / np.log(self.max_iter + 1)
        return max(min(alpha, self.A), self.B)
    def iterative_decay(self, current_iter):
        if current_iter >= self.max_iter:
            return 1
        else:
            return 0

from utils.general_utils import build_scaling_rotation
def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    #symm = strip_symmetric(actual_covariance)
    #return symm
    return actual_covariance
def compute_density(gaussian_centers, grid_point, density, covariance, expand=[5,15,15]):
    # grid_point: [1, z, x, y, 1, 3]
    z, x, y = grid_point.shape[1:4]
    num_gaussians = gaussian_centers.shape[-2]
    # initialize density_grid outside the loop
    density_grid = torch.zeros(1, z, x, y, 1, device='cuda')
    expanded_grid_point = grid_point.expand(num_gaussians, z, x, y, 1, 3)

    mean_zxy = gaussian_centers * torch.tensor([z, x, y]).cuda()  # [num_gaussian, 3]
    mean_z, mean_x, mean_y = mean_zxy[:, 0], mean_zxy[:, 1], mean_zxy[:, 2]
    # 在计算距离时,索引num_gaussians_in_batch个小patch,而不是整个大patch, 与每个gaussian_center分别计算距离
    # pytorch只支持索引统一大小的patch,因此使用固定大小
    z_indices = torch.clamp((mean_z.unsqueeze(-1) - expand[0] / 2).int() + torch.arange(0, expand[0], device='cuda'), 0,
                            z - 1)  # [num_gaussian_in_patch, expand[0]]
    x_indices = torch.clamp((mean_x.unsqueeze(-1) - expand[1] / 2).int() + torch.arange(0, expand[1], device='cuda'), 0,
                            x - 1)  # [num_gaussian_in_patch, expand[1]]
    y_indices = torch.clamp((mean_y.unsqueeze(-1) - expand[2] / 2).int() + torch.arange(0, expand[2], device='cuda'), 0,
                            y - 1)  # [num_gaussian_in_patch, expand[2]]

    grid_indices = torch.arange(num_gaussians, device='cuda').view(-1, 1, 1, 1)
    z_indices = z_indices.view(num_gaussians, -1, 1, 1)  # [num_gaussians_in_batch, expand[0], 1, 1]
    x_indices = x_indices.view(num_gaussians, 1, -1, 1)  # [num_gaussians_in_batch, 1, expand[1], 1]
    y_indices = y_indices.view(num_gaussians, 1, 1, -1)  # [num_gaussians_in_batch, 1, 1, expand[2]]
    patches = expanded_grid_point[grid_indices, z_indices, x_indices, y_indices, :,:]  # [num_gaussians_in_batch, expand[0], expand[1], expand[2], 1, 3]

    # dist = torch.norm(patches - gaussian_centers.view(num_gaussians, 1,1,1,1, 3), dim=-1) # [num_gaussian_in_patch, expand[0], expand[1], expand[2], 1]

    # density_patch = (density.view(-1, 1, 1, 1, 1) * torch.exp(-dist**2 / (2*sigma.view(-1, 1, 1, 1, 1)**2))) # [num_gaussian_in_patch, expand[0], expand[1], expand[2], 1]
    regularization_term = 1e-6 * torch.eye(3, device='cuda')
    regularized_covariance = covariance + regularization_term
    density_patch = (density.view(-1, 1, 1, 1, 1) * torch.exp(-0.5 * torch.matmul(torch.matmul((patches - gaussian_centers.view(num_gaussians, 1, 1, 1, 1, 3)).unsqueeze(-2),
                     torch.inverse(regularized_covariance.view(num_gaussians, 1, 1, 1, 1, 3, 3))),(patches - gaussian_centers.view(num_gaussians, 1, 1, 1, 1, 3)).unsqueeze(-1)).squeeze(-1).squeeze(-1)))  # [num_gaussian_in_patch, expand[0], expand[1], expand[2], 1]
    # Prepare indices for adding the patch back to the density_grid
    indices = ((z_indices * x + x_indices) * y + y_indices).view(-1)
    # Add the density patch back to the density_grid
    density_grid = density_grid.view(-1)  # [1*z*x*y*1]
    density_patch = density_patch.view(-1)  # [num_gaussians_in_batch*expand[0]*expand[1]*expand[2]*1]

    density_grid.scatter_add_(0, indices, density_patch)
    # density_grid = density_grid.scatter_add(0, indices, density_patch)
    # Reshape density_grid back to its original shape
    density_grid = density_grid.view(1, z, x, y, 1)

    return density_grid

def volume_render(gaussian_splats, model_cfg):
    grid = create_grid_3d(model_cfg.data.training_resolution, model_cfg.data.training_resolution, model_cfg.data.training_resolution)
    grid = grid.cuda()
    # train_data[0] grid: [batchsize, z, x, y, 3]
    grid = grid.unsqueeze(0).repeat(model_cfg.opt.batch_size, 1, 1, 1, 1)
    grid_expanded = grid.unsqueeze(-2)
    # train_output = gaussians.grid_sample(grid, expand=[5, 15, 15])
    covariance = build_covariance_from_scaling_rotation(gaussian_splats["_scaling"], 1.0,gaussian_splats["_rotation"])
    output = compute_density(gaussian_splats["_xyz"], grid_expanded, gaussian_splats["_density"], covariance, expand=[5, 15, 15])
    return output
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def img_tv_loss(img):
    b, c, h, w = img.size()
    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
    tv_z = torch.abs(img[:, 1:, :, :] - img[:, :-1, :, :]).sum()
    return (tv_h + tv_w + tv_z) / (b * c * h * w)

def volume2pointsplusplus(volume):
    # 假设 volume 形状为 [1, depth, height, width]
    volume = volume.squeeze(0)
    points = torch.argwhere(volume)
    values = volume[points[:, 0], points[:, 1], points[:, 2]].unsqueeze(-1)
    pointsplusplus = torch.cat((points, values), dim=1)
    return pointsplusplus


def volume2pointsplusplus_old(volume):
    # 假设 volume 形状为 [1, depth, height, width]
    depth, height, width = volume.shape[1:]

    # 生成坐标网格
    z, y, x = torch.meshgrid(torch.arange(depth),
                             torch.arange(height),
                             torch.arange(width))

    # 展平网格和体素数据
    z, y, x = z.flatten(), y.flatten(), x.flatten()
    values = volume.flatten()
    # 筛选出非零的体素及其坐标
    mask = values != 0
    z, y, x, values = z[mask], y[mask], x[mask], values[mask]
    # 将坐标和值组合为点云数据
    pointsplusplus = torch.stack((x, y, z, values), dim=1)

    return pointsplusplus
# 传入的点云坐标范围0-1
def pointsplusplus2volume(pointsplusplus, volume_size=128):
    # 确保点的坐标部分被转换为整数索引
    pointsplusplus[:, :3] = (pointsplusplus[:, :3] * (volume_size - 1)).long()

    # 创建一个全零的体素数据
    volume = torch.zeros(1, volume_size, volume_size, volume_size, dtype=pointsplusplus.dtype, device=pointsplusplus.device)

    # 将点云数据分解为坐标和值
    coordinates = pointsplusplus[:, :3].long()
    values = pointsplusplus[:, 3]

    # 使用高级索引一次性更新体素数据
    volume[0, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = values

    return volume

def voxel_to_depth_map_differentiable(voxel_data):
    # 假设体素数据的形状为(1, D, H, W)，先去掉批次维度
    voxel_data = voxel_data.squeeze(0)

    # 计算y轴方向的累积最大值
    cumulative_max, _ = voxel_data.cummax(dim=1)

    # 计算深度图，这里使用了cumulative_max最后一个非零值的位置作为深度值
    depth_map = (cumulative_max > 0).float().sum(dim=1) / voxel_data.shape[1]

    return depth_map

def rotate_pointsplusplus(points, theta):
    # if axis == 'x':
    #     R = rotation_matrix_x(theta)
    # elif axis == 'y':
    #     R = rotation_matrix_y(theta)
    # elif axis == 'z':
    R = rotation_matrix_z(theta)
    # 仅旋转坐标部分（前三列）
    return torch.cat((torch.mm(points[:, :3], R.t()), points[:, 3].unsqueeze(1)), dim=1)
def voxel_to_depth_map(voxel_data):
    """
    将3D体素数据转换为沿y轴方向的深度图。

    参数:
    - voxel_data: 3D体素数据，尺寸为(1, x, y, z)，假设大部分值为0，非零值代表某种结构。

    返回:
    - depth_map: 深度图，尺寸为(x,y)，代表从y轴负方向向正方向看去的结构深度。
    """
    # 假设体素数据的形状为(1, D, H, W)，先去掉批次维度
    voxel_data = voxel_data.squeeze(0)

    # 初始化深度图为全0，尺寸为(H, W)
    depth_map = torch.ones(voxel_data.shape[0], voxel_data.shape[2])

    # 遍历x-z平面上的每个位置
    for x in range(voxel_data.shape[0]):
        for z in range(voxel_data.shape[2]):
            # 在y轴方向查找第一个非零体素
            for y in range(voxel_data.shape[1]):
                if voxel_data[x, y, z] > 0:
                    # 计算非零体素到x0z平面的距离，即y的值
                    depth_map[x, z] = y/voxel_data.shape[1]
                    break  # 找到第一个非零值后，跳出循环

    return depth_map

def voxel_to_depth_map_differentiable(voxel_data):
    # 假设体素数据的形状为(1, D, H, W)，先去掉批次维度
    voxel_data = voxel_data.squeeze(0)
    # 计算y轴方向的累积最大值
    cumulative_max, _ = voxel_data.cummax(dim=1)
    # 计算深度图，这里使用了cumulative_max最后一个非零值的位置作为深度值
    depth_map = (cumulative_max > 0).float().sum(dim=1) / voxel_data.shape[1]
    return depth_map

def voxel_to_depth_map_differentiable_2(voxel_data):
    # 假设体素数据的形状为(1, D, H, W)，先去掉批次维度
    voxel_data = voxel_data.squeeze(0)
    # 计算y轴方向的累积最大值
    cumulative_max, _ = voxel_data.cummax(dim=0)
    # 计算深度图，这里使用了cumulative_max最后一个非零值的位置作为深度值
    depth_map = (cumulative_max > 0).float().sum(dim=1) / voxel_data.shape[1]
    return depth_map

def point_cloud_to_depth_map(points, depth_map_size=128):
    # 使用torch操作初始化深度图，所有值设置为最大（1表示）
    depth_map = torch.ones((depth_map_size, depth_map_size), dtype=torch.float32)

    # 规范化点云的x和z坐标到[0, depth_map_size-1]，以便映射到深度图
    x_indices = torch.floor(points[:, 0] * (depth_map_size - 1)).to(torch.int64)
    z_indices = torch.floor(points[:, 2] * (depth_map_size - 1)).to(torch.int64)

    # 为了找到每个(x, z)位置上y的最小值，我们先创建一个空的字典来存储信息
    min_y_dict = {}

    # 遍历所有点，更新(x, z)位置的最小y值
    for i in range(points.size(0)):
        x = x_indices[i].item()
        z = z_indices[i].item()
        y = points[i, 1].item()
        if (x, z) not in min_y_dict or y < min_y_dict[(x, z)]:
            min_y_dict[(x, z)] = y

    # 使用计算出的最小y值更新深度图
    for (x, z), y in min_y_dict.items():
        depth_map[x, z] = y

    return depth_map

def resample_volume(image, target_shape=(128, 128, 128)):
    # 从原始体素数据中提取形状信息
    x,y,z = image.shape
    pos = np.where(image>0)
    x_min, x_max = pos[0].min(), pos[0].max()
    y_min, y_max = pos[1].min(), pos[1].max()
    z_min, z_max = pos[2].min(), pos[2].max()
    leng = 120
    leng_half = leng // 2
    x_range = x_max - x_min + leng
    y_range = y_max - y_min + leng
    z_range = z_max - z_min + leng
    # 计算每个维度的缩放比例
    x_scale = target_shape[0] / x_range
    y_scale = target_shape[1] / y_range
    z_scale = target_shape[2] / z_range
    # 生成新的体素数据
    if x_min-leng_half < 0 or y_min-leng_half < 0 or z_min-leng_half < 0 or x_max+leng_half > x or y_max+leng_half > y or z_max+leng_half > z:
        print("Abnormal data")
    image_v = image[x_min-leng_half:x_max+leng_half, y_min-leng_half:y_max+leng_half, z_min-leng_half:z_max+leng_half]
    resampled_image = zoom(image_v, (x_scale, y_scale, z_scale), order=1)
    resampled_image[resampled_image < 0] = 0
    return resampled_image

if __name__ == '__main__':
    import json

    cpu_num = 10
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    import pandas as pd
    import argparse
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Pass two numbers")
    #
    # # Add the parameters
    parser.add_argument('--n1', type=int, default= 0,  help="First number")
    parser.add_argument('--n2', type=int, default= 10, help="Second number")
    args = parser.parse_args()

    # read json file to split the dataset
    # with open(r"CTData.json", 'r') as load_f:
    #     load_dict = json.load(load_f)
    #     # train_list = load_dict['train']
    #     val_list = load_dict['validation']
    #     test_list = load_dict['test']
    # Alldata_list =  val_list + test_list

    scale_size = 128
    batch_size = 1
    num_proj = 16
    image_size = [128] * 3
    proj_size = [128] * 3
    ct_projector = ConeBeam3DProjector(image_size, proj_size, num_proj)


    scale_size = 128
    batch_size = 1
    num_proj = 16
    image_size = [128] * 3
    proj_size = [128] * 3
    ct_projector = ConeBeam3DProjector(image_size, proj_size, num_proj)
    # ct_projector.raw_reso = 0.1
    CASDataset_path = '/media/F/yyk/Dataset/imageCAS'
    df = pd.read_excel("/media/I/xcw/3DGR-CAR-main/3dgs-car/imageCAS_data_split.xlsx")
    CASData_list = df.iloc[:, 0].tolist()
    # remove the "FileName" in the list
    CASData_list.remove("FileName")

    save_dir = "/media/I/xcw/3DGR-CAR-main/all_data"
    #image_dir = r"/data/xuemingfu/project1/Coronary_Angiography_data/ImageCAS/dataset"
    image_dir = "/media/F/yyk/Dataset/imageCAS"
    for file in CASData_list[args.n1:args.n2]:
        print("filename: ", file)
        file_path = os.path.join(image_dir, str(file), 'img.nii.gz')
        label_path = os.path.join(image_dir, str(file), 'label.nii.gz')
        image = nib.load(file_path)
        labels = nib.load(label_path)
        image_mha = image.get_fdata() * (labels.get_fdata() == 1)
        image_mha[image_mha < 0] = 0

        image_mha = crop_to_standard_cas(image_mha, scale=480)  # ImageCAS
        labels_cor = crop_to_standard_cas(labels.get_fdata(), scale=480)  # ImageCAS
        image_mha = resample_valid_voxels_to_standard(image_mha, 128)
        labels_cor = resample_valid_voxels_to_standard(labels_cor , 128)

        mask = labels_cor == 0
        image_mha = torch.tensor(image_mha, dtype=torch.float32)[None, ...]
        mask = torch.tensor(mask, dtype=torch.float32)[None, ...]
        image_mha = F.interpolate(image_mha, size=(128, 128), mode='bilinear', align_corners=False)
        resampled_mask = F.interpolate(mask, size=(128, 128), mode='bilinear', align_corners=False)
        image_mha[resampled_mask.bool()] = 0

        image_mha = image_mha / torch.max(image_mha)  # [1, C, H, W]

        # image_mha = image_mha.permute(1, 2, 3, 0)  # [C, H, W, 1]
        # image_mha = image_mha.unsqueeze(0)  #[1, C, H, W, 1]
        # image_mha = image_mha.transpose(1, 4).squeeze(1)
        # image_mha = image_mha.transpose(1, 3)

        # resampled_mask = resampled_mask.permute(1, 2, 3, 0)  # [C, H, W, 1]
        # resampled_mask = resampled_mask.unsqueeze(0)  # [1, C, H, W, 1]
        # resampled_mask = resampled_mask.transpose(1, 4).squeeze(1)
        # resampled_mask = resampled_mask.transpose(1, 3)

        voxel_center = torch.tensor([64,64,64], dtype=torch.float32)

        #label
        points = torch.argwhere(1 - resampled_mask.squeeze(0))
        points_voxels = 1 - resampled_mask.squeeze(0)
        #centerline   始终保持物体和物体的点云是一致的   投影方向不变，只取读一个
        # centerline_voxels = torch.tensor(skeletonize_3d(points_voxels)).long()
        # --- 3D骨架化 ---
        points_voxels_np = points_voxels.detach().cpu().numpy().astype(bool)
        skeleton_np = skeletonize_3d(points_voxels_np)
        centerline_voxels = torch.from_numpy(skeleton_np.astype(np.int64)).to(points_voxels.device)
        
        centerline_points = np.argwhere(centerline_voxels).T
        number = 360
        bg_mask = torch.zeros(number , points.shape[0],3)
        depthes = torch.zeros(number , 128,128)
        cl_mask = torch.zeros(number , centerline_points.shape[0],3)
        projs_all = torch.zeros(number , 128, 128)
        points_centered = points - voxel_center
        centerline_points_centered = centerline_points - voxel_center
        #旋转resampled_mask 32次
        #第一次旋转
        points_image = volume2pointsplusplus(image_mha)
        points_image_center = points_image[:,:3] - voxel_center
        points_value = points_image[:, 3].unsqueeze(1)
        volume = torch.zeros(number , 1, 128, 128, 128, dtype=points_image.dtype, device=points_image.device)
        from multiprocessing import Pool
        def process(i):
            # 首先旋转点云
            bg_mask = (rotate_coordinate_system(points_centered, i) + voxel_center) / 128
            cl_mask = (rotate_coordinate_system(centerline_points_centered, i) + voxel_center) / 128
            # 将旋转后的点云转回体素
            points_image_rotated = rotate_coordinate_system(points_image_center, i) + voxel_center
            points_image_rotated = torch.cat((points_image_rotated / 128, points_value), dim=1)
            rotated_image = pointsplusplus2volume(points_image_rotated, 128)
            projs = ct_projector.forward_project(rotated_image).cpu()
            print(projs.shape)
            projs_all = torch.flip(projs.squeeze(0)[0, :, :], [1]).T
            dep = voxel_to_depth_map(rotated_image)
            depthes = torch.flip(dep, [1]).T
            volume = rotated_image
            return projs_all, bg_mask, cl_mask, depthes, volume


        with Pool() as p:
            results = p.map(process, range(number))

        projs_all, bg_mask, cl_mask, depthes, volume = zip(*results)

        filename = os.path.join(save_dir, str(file) + ".pt")
        filename2 = os.path.join(save_dir, str(file) + "_volume.pt")
        torch.save({'projs': projs_all, 'bg_mask': bg_mask, 'cl_mask': cl_mask, 'depth': depthes}, filename)
        torch.save({'volume': volume}, filename2)

        # 将数据保存为 .nii.gz 文件（假设 volume 数据转换为 nii.gz）
        nii_file = os.path.join(save_dir, str(file) + "_volume.nii.gz")
        volume_data = torch.cat(volume).numpy()
        nii_image = nib.Nifti1Image(volume_data, np.eye(4))  # 假设使用单位矩阵作为仿射矩阵
        nib.save(nii_image, nii_file)

        # 删除源 .pt 文件
        os.remove(filename)
        os.remove(filename2)