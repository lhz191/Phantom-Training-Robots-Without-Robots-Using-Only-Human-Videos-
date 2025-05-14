import numpy as np
import torch
import pyrender
import trimesh
import cv2
from typing import List, Tuple, Optional, Dict
import os

class RobotRenderer:
    """
    机器人渲染器类，用于将机器人模型渲染到图像中
    """
    def __init__(
        self,
        robot_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化渲染器
        
        Args:
            robot_model_path: 机器人3D模型文件路径(.obj或.stl格式)
            device: 计算设备
        """
        self.device = device
        
        # 加载机器人模型
        if os.path.exists(robot_model_path):
            self.robot_mesh = trimesh.load(robot_model_path)
        else:
            raise FileNotFoundError(f"Robot model not found at {robot_model_path}")
            
        # 渲染参数
        self.light_intensity = 3.0
        self.light_color = np.array([1.0, 1.0, 1.0])
        
    def create_raymond_lights(self) -> List[pyrender.Node]:
        """创建Raymond lighting setup"""
        nodes = []

        # 创建3个定向光源
        for i, direct in enumerate([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]):
            direct = np.array(direct)
            pos = direct + np.array([0, 0, 0])
            
            # 创建定向光源
            light = pyrender.DirectionalLight(
                color=self.light_color,
                intensity=self.light_intensity
            )
            
            # 创建光源节点
            light_node = pyrender.Node(
                light=light,
                matrix=np.eye(4)
            )
            nodes.append(light_node)

        return nodes
        
    def render_robot(
        self,
        pt: np.ndarray,  # (3,) 末端执行器位置
        Rt: np.ndarray,  # (6,) 末端执行器方向
        gt: float,       # 夹持器开度
        img_size: Tuple[int, int],  # 图像尺寸 (H, W)
        camera_params: Dict,  # 相机参数
        background_img: Optional[np.ndarray] = None  # 背景图像
    ) -> np.ndarray:
        """
        渲染机器人到图像中
        
        Args:
            pt: 末端执行器位置
            Rt: 末端执行器方向(6D rotation representation)
            gt: 夹持器开度(0-1)
            img_size: 输出图像尺寸
            camera_params: 相机参数字典，包含:
                - focal_length: 焦距
                - camera_center: 相机中心点 [cx, cy]
                - camera_rotation: 相机旋转矩阵 (3, 3)
                - camera_translation: 相机平移向量 (3,)
            background_img: 可选的背景图像
            
        Returns:
            渲染后的图像 (H, W, 3)
        """
        H, W = img_size
        
        # 创建离屏渲染器
        renderer = pyrender.OffscreenRenderer(
            viewport_width=W,
            viewport_height=H
        )
        
        # 创建场景
        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],
            ambient_light=(0.3, 0.3, 0.3)
        )
        
        # 创建机器人网格
        robot_mesh = self.robot_mesh.copy()
        
        # 应用变换
        # 1. 根据夹持器开度调整mesh
        # TODO: 根据具体机器人模型实现夹持器开合的mesh变形
        
        # 2. 应用位姿变换
        transform = np.eye(4)
        transform[:3, :3] = self.rotation_6d_to_matrix(Rt)
        transform[:3, 3] = pt
        robot_mesh.apply_transform(transform)
        
        # 创建材质
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(0.7, 0.7, 0.7, 1.0)
        )
        
        # 添加机器人mesh到场景
        mesh = pyrender.Mesh.from_trimesh(
            robot_mesh,
            material=material
        )
        scene.add(mesh, 'robot')
        
        # 添加相机
        camera = pyrender.IntrinsicsCamera(
            fx=camera_params['focal_length'],
            fy=camera_params['focal_length'],
            cx=camera_params['camera_center'][0],
            cy=camera_params['camera_center'][1]
        )
        
        # 设置相机位姿
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = camera_params['camera_rotation']
        camera_pose[:3, 3] = camera_params['camera_translation']
        scene.add(camera, pose=camera_pose)
        
        # 添加光源
        light_nodes = self.create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)
            
        # 渲染
        color, depth = renderer.render(
            scene,
            flags=pyrender.RenderFlags.RGBA
        )
        
        # 转换为float32并归一化
        color = color.astype(np.float32) / 255.0
        
        # 如果有背景图像，进行混合
        if background_img is not None:
            # 确保背景图像是float32且归一化
            background_img = background_img.astype(np.float32) / 255.0
            if background_img.shape[-1] == 3:
                background_img = np.concatenate([
                    background_img,
                    np.ones_like(background_img[:, :, :1])
                ], axis=-1)
            
            # 使用alpha通道进行混合
            alpha = color[:, :, 3:4]
            color_rgb = color[:, :, :3]
            background_rgb = background_img[:, :, :3]
            
            blended = color_rgb * alpha + background_rgb * (1 - alpha)
            
            # 转回uint8
            return (blended * 255).astype(np.uint8)
        else:
            # 转回uint8
            return (color[:, :, :3] * 255).astype(np.uint8)
            
    @staticmethod
    def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
        """
        将6D旋转表示转换为3x3旋转矩阵
        
        Args:
            rotation_6d: (6,) 6D旋转表示
            
        Returns:
            (3, 3) 旋转矩阵
        """
        # 将6D向量重塑为两个3D向量
        v1 = rotation_6d[:3]
        v2 = rotation_6d[3:]
        
        # 标准化第一个向量
        v1 = v1 / np.linalg.norm(v1)
        
        # 计算第二个向量的正交分量
        v2_orthogonal = v2 - np.dot(v1, v2) * v1
        v2_orthogonal = v2_orthogonal / np.linalg.norm(v2_orthogonal)
        
        # 计算第三个正交向量
        v3 = np.cross(v1, v2_orthogonal)
        
        # 组合成旋转矩阵
        R = np.stack([v1, v2_orthogonal, v3], axis=1)
        
        return R 