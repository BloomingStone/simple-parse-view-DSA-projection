from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectionAngles:
    alpha_beta: np.ndarray  # shape: (N, 2), in radians
    
    @staticmethod
    def from_alpha_beta_list(angles: list[tuple[float, float]]) -> "ProjectionAngles":
        """从角度列表创建 ProjectionAngles，角度以度为单位。"""
        angles_array = np.array(angles)
        if angles_array.ndim != 2 or angles_array.shape[1] != 2:
            raise ValueError("angles must be a list of (alpha, beta) pairs")
        return ProjectionAngles(np.deg2rad(angles_array))
    
    @staticmethod
    def from_alpha_beta_arrays(alphas: np.ndarray, betas: np.ndarray) -> "ProjectionAngles":
        """从 alpha 和 beta 数组创建 ProjectionAngles，角度以度为单位。"""
        if alphas.shape != betas.shape:
            raise ValueError("alphas and betas must have the same shape")
        return ProjectionAngles(np.column_stack((np.deg2rad(alphas), np.deg2rad(betas))))
    
    @staticmethod
    def from_random(num_angles: int, seed: int = 42) -> "ProjectionAngles":
        """生成随机 alpha/beta 角度对，角度以度为单位。"""
        rng = np.random.default_rng(seed)
        alphas = rng.uniform(-60, 60, num_angles)  # alpha 范围 [-60, 60] 度
        betas = rng.uniform(-30, 30, num_angles)   # beta 范围 [-30, 30] 度
        return ProjectionAngles.from_alpha_beta_arrays(alphas, betas)
    
    @property
    def alphas(self) -> np.ndarray:
        return self.alpha_beta[:, 0]
    
    @property
    def betas(self) -> np.ndarray:
        return self.alpha_beta[:, 1]
    
    @property
    def num_angles(self) -> int:
        return self.alpha_beta.shape[0]
    
    def __iter__(self):
        return iter(self.alpha_beta)
    
    @property
    def meta(self) -> dict:
        """返回角度信息的元数据字典。"""
        return {
            "num_angles": self.num_angles,
            "alphas": self.alphas.tolist(),
            "betas": self.betas.tolist(),
        }
