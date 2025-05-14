import numpy as np
import json
import os
from scipy.spatial import ConvexHull
from scipy.linalg import svd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def apply_icp_transformation(keypoints_3d, transformation_matrix):
    """
    将ICP转换矩阵应用到关键点上
    
    参数:
        keypoints_3d: numpy array, shape (21, 3), HaMeR的21个关键点
        transformation_matrix: numpy array, shape (4, 4), ICP转换矩阵
        
    返回:
        transformed_keypoints: numpy array, shape (21, 3), 转换后的关键点
    """
    # 将关键点转换为齐次坐标
    ones = np.ones((keypoints_3d.shape[0], 1))
    keypoints_homo = np.concatenate([keypoints_3d, ones], axis=1)  # (21, 4)
    
    # 应用转换矩阵
    transformed_keypoints_homo = (transformation_matrix @ keypoints_homo.T).T  # (21, 4)
    
    # 转换回3D坐标
    transformed_keypoints = transformed_keypoints_homo[:, :3]
    
    return transformed_keypoints

def constrain_finger_joints(keypoints_3d):
    """
    限制拇指和食指最后两个关节为单自由度
    
    参数:
        keypoints_3d: numpy array, shape (21, 3), 关键点
        
    返回:
        constrained_keypoints: numpy array, shape (21, 3), 约束后的关键点
    """
    # 复制关键点以避免修改原始数据
    constrained_keypoints = keypoints_3d.copy()
    
    # 拇指关节索引
    thumb_joints = [1, 2, 3, 4]  # 从根部到指尖
    # 食指关节索引
    index_joints = [5, 6, 7, 8]  # 从根部到指尖
    
    # 为拇指和食指分别处理
    for finger_joints in [thumb_joints, index_joints]:
        # 1. 确定弯曲平面 - 使用前三个关节点
        p1 = keypoints_3d[finger_joints[0]]  # 根部
        p2 = keypoints_3d[finger_joints[1]]  # 第二关节
        p3 = keypoints_3d[finger_joints[2]]  # 第三关节
        
        # 计算平面法向量（弯曲轴）
        v1 = p2 - p1
        v2 = p3 - p2
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) > 1e-8:  # 避免零向量
            normal = normal / np.linalg.norm(normal)  # 单位化
        else:
            # 如果无法确定平面（三点共线），使用默认法向量
            normal = np.array([0, 0, 1])
        
        # 2. 处理最后两个关节
        for i in range(2, 4):  # 最后两个关节
            prev_joint = finger_joints[i-1]
            curr_joint = finger_joints[i]
            
            if i < 3:  # 倒数第二关节 (i=2)
                next_joint = finger_joints[i+1]
                
                # 计算当前角度
                v1 = keypoints_3d[curr_joint] - keypoints_3d[prev_joint]
                v2 = keypoints_3d[next_joint] - keypoints_3d[curr_joint]
                
                # 确保向量非零
                if np.linalg.norm(v1) < 1e-8 or np.linalg.norm(v2) < 1e-8:
                    continue
                
                # 投影向量到弯曲平面
                v1_proj = v1 - np.dot(v1, normal) * normal
                v2_proj = v2 - np.dot(v2, normal) * normal
                
                # 确保投影向量非零
                if np.linalg.norm(v1_proj) < 1e-8 or np.linalg.norm(v2_proj) < 1e-8:
                    continue
                
                # 计算投影后的角度
                cos_angle = np.dot(v1_proj, v2_proj) / (np.linalg.norm(v1_proj) * np.linalg.norm(v2_proj))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
                angle = np.arccos(cos_angle)
                
                # 限制角度范围
                max_angle = np.pi / 2  # 90度
                if angle > max_angle:
                    # 构建旋转轴（垂直于弯曲平面内的两个向量）
                    rotation_axis = np.cross(v1_proj, v2_proj)
                    if np.linalg.norm(rotation_axis) < 1e-8:
                        continue
                    
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    rotation_angle = -(angle - max_angle)  # 需要减小角度
                    
                    # 创建罗德里格斯旋转公式
                    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                  [rotation_axis[2], 0, -rotation_axis[0]],
                                  [-rotation_axis[1], rotation_axis[0], 0]])
                    R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)
                    
                    # 调整next_joint位置
                    v2_rotated = R @ v2
                    constrained_keypoints[next_joint] = keypoints_3d[curr_joint] + v2_rotated
            
            else:  # 指尖关节 (i=3)
                # 对于指尖，确保它与前一关节的连线在弯曲平面内
                # 使用已更新的constrained_keypoints而不是原始的keypoints_3d
                v = constrained_keypoints[curr_joint] - keypoints_3d[prev_joint]
                
                if np.linalg.norm(v) < 1e-8:
                    continue
                
                # 投影到弯曲平面
                v_proj = v - np.dot(v, normal) * normal
                
                if np.linalg.norm(v_proj) < 1e-8:
                    continue
                
                # 保持长度不变，但方向调整到平面内
                v_new = v_proj / np.linalg.norm(v_proj) * np.linalg.norm(v)
                
                # 调整指尖位置
                constrained_keypoints[curr_joint] = keypoints_3d[prev_joint] + v_new
                
                # 额外限制：确保与前一段的角度不超过90度
                prev_prev_joint = finger_joints[i-2]
                # 使用已更新的constrained_keypoints计算v2
                v1 = keypoints_3d[prev_joint] - keypoints_3d[prev_prev_joint]
                v2 = constrained_keypoints[curr_joint] - keypoints_3d[prev_joint]
                
                if np.linalg.norm(v1) < 1e-8 or np.linalg.norm(v2) < 1e-8:
                    continue
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                max_angle = np.pi / 2
                if angle > max_angle:
                    # 简化处理：直接在平面内调整角度
                    # 保存v2的长度
                    v2_length = np.linalg.norm(v2)
                    
                    # 创建垂直于v1的单位向量（在已确定的平面内）
                    # 首先确保v1是单位向量
                    v1_unit = v1 / np.linalg.norm(v1)
                    # 使用平面法向量和v1创建垂直于v1的向量
                    perpendicular = np.cross(normal, v1_unit)
                    if np.linalg.norm(perpendicular) > 1e-8:
                        perpendicular = perpendicular / np.linalg.norm(perpendicular)
                        
                        # 确保选择的是90度方向而不是270度方向
                        # 检查垂直向量与原始v2的点积，如果为负，说明夹角>90度，需要反转
                        if np.dot(perpendicular, v2) < 0:
                            perpendicular = -perpendicular
                    else:
                        # 如果遇到数值问题，使用简单的90度旋转
                        continue
                    
                    # 调整v2为与v1垂直的向量（在平面内），保持长度不变
                    v2_new = perpendicular * v2_length
                    
                    # 更新指尖位置
                    constrained_keypoints[curr_joint] = keypoints_3d[prev_joint] + v2_new
    
    return constrained_keypoints

def compute_robot_action(keypoints_3d):
    """
    从HaMeR关键点计算机器人动作
    
    参数:
        keypoints_3d: numpy array, shape (21, 3), HaMeR的21个关键点
        
    返回:
        pt: numpy array, shape (3,), 末端执行器位置
        Rt: numpy array, shape (6,), 末端执行器方向（6D连续旋转表示）
        gt: float, 夹持器开度（0-1之间）
    """
    # 获取拇指和食指的关键点索引
    # 注意：这里的索引需要根据HaMeR的关键点定义来调整
    thumb_tip_idx = 4  # 拇指尖
    index_tip_idx = 8  # 食指尖
    
    # 获取拇指和食指的所有关键点
    thumb_indices = [1, 2, 3, 4]  # 拇指关键点，不包含0（手掌中心）
    index_indices = [5, 6, 7, 8]     # 食指关键点
    
    thumb_points = keypoints_3d[thumb_indices]
    index_points = keypoints_3d[index_indices]
    
    # 1. 计算位置 pt（拇指和食指指尖的中点）
    pt = (keypoints_3d[thumb_tip_idx] + keypoints_3d[index_tip_idx]) / 2
    
    # 2. 计算方向 Rt
    # 2.1 将所有拇指和食指关键点组合用于平面拟合
    all_points = np.vstack([thumb_points, index_points])
    
    # 2.2 计算这些点的协方差矩阵
    centered_points = all_points - np.mean(all_points, axis=0)
    cov_matrix = np.dot(centered_points.T, centered_points)
    
    # 2.3 使用SVD计算平面法向量
    U, S, Vh = svd(cov_matrix)
    
    # 2.4 获取平面的法向量（最小特征值对应的方向）
    normal = U[:, 2]  # 平面的法向量
    
    # 2.5 确保法向量方向一致
    if np.dot(normal, pt - np.mean(all_points, axis=0)) < 0:
        normal = -normal
    
    # 2.6 通过拇指关键点拟合主轴向量（按论文要求）
    # 使用拇指从根部到指尖的方向作为主轴
    principal_axis = thumb_points[-1] - thumb_points[0]  # 拇指尖 - 拇指根
    
    # 归一化主轴向量
    if np.linalg.norm(principal_axis) > 1e-8:
        principal_axis = principal_axis / np.linalg.norm(principal_axis)
    else:
        # 如果主轴长度接近零，使用默认方向
        principal_axis = np.array([1, 0, 0])
    
    # 2.7 确保主轴与法向量垂直（通过投影去除法向量分量）
    # 这是为了保证旋转表示的有效性
    principal_axis = principal_axis - np.dot(principal_axis, normal) * normal
    
    # 再次归一化
    if np.linalg.norm(principal_axis) > 1e-8:
        principal_axis = principal_axis / np.linalg.norm(principal_axis)
    else:
        # 如果主轴与法向量几乎平行，构造一个垂直向量
        principal_axis = np.array([1, 0, 0]) - np.dot(np.array([1, 0, 0]), normal) * normal
        principal_axis = principal_axis / np.linalg.norm(principal_axis)
    
    # 2.8 将方向转换为6D连续旋转表示
    # 根据论文描述，使用法向量和主轴向量定义方向
    Rt = np.concatenate([normal, principal_axis])
    
    # 3. 计算夹持器开度 gt
    # 3.1 计算指尖距离
    finger_distance = np.linalg.norm(keypoints_3d[thumb_tip_idx] - keypoints_3d[index_tip_idx])
    
    # 3.2 归一化到0-1范围
    # 注意：这里的最大距离需要根据实际情况调整
    max_distance = 0.1  # 假设最大距离为10厘米
    gt = np.clip(finger_distance / max_distance, 0, 1)
    
    return pt, Rt, gt

def process_hamer_output(keypoints_path, icp_info_path):
    """
    处理HaMeR输出文件，计算机器人动作
    
    参数:
        keypoints_path: str, HaMeR输出文件的路径（JSON格式）
        icp_info_path: str, ICP配准信息的路径（JSON格式）
        
    返回:
        action: dict, 机器人动作
    """
    # 加载HaMeR输出
    with open(keypoints_path, 'r') as f:
        hamer_data = json.load(f)
    
    # 加载ICP配准信息
    with open(icp_info_path, 'r') as f:
        icp_info = json.load(f)
    
    # 获取关键点数据（优先使用transformed关键点）
    if 'keypoints_3d_transformed' in hamer_data:
        keypoints_3d = np.array(hamer_data['keypoints_3d_transformed'])
    else:
        keypoints_3d = np.array(hamer_data['keypoints_3d'])
    
    # 应用ICP转换矩阵
    transformation_matrix = np.array(icp_info['transformation_matrix'])
    transformed_keypoints = apply_icp_transformation(keypoints_3d, transformation_matrix)
    
    # 约束手指关节
    constrained_keypoints = constrain_finger_joints(transformed_keypoints)
    
    # 计算机器人动作
    pt, Rt, gt = compute_robot_action(constrained_keypoints)
    
    # 构建动作字典
    action = {
        'position': pt.tolist(),
        'orientation': Rt.tolist(),
        'gripper': float(gt),
        'original_keypoints': keypoints_3d.tolist(),
        'transformed_keypoints': transformed_keypoints.tolist(),
        'constrained_keypoints': constrained_keypoints.tolist()
    }
    
    return action

def visualize_keypoints(original_keypoints, transformed_keypoints, constrained_keypoints, output_dir):
    """
    可视化关键点的变换过程
    
    参数:
        original_keypoints: numpy array, shape (21, 3), 原始关键点
        transformed_keypoints: numpy array, shape (21, 3), ICP转换后的关键点
        constrained_keypoints: numpy array, shape (21, 3), 关节约束后的关键点
        output_dir: str, 输出目录
    """
    # 定义手部关键点连接关系
    connections = [
        # 拇指
        [0, 1], [1, 2], [2, 3], [3, 4],
        # 食指
        [0, 5], [5, 6], [6, 7], [7, 8],
        # 中指
        [0, 9], [9, 10], [10, 11], [11, 12],
        # 无名指
        [0, 13], [13, 14], [14, 15], [15, 16],
        # 小指
        [0, 17], [17, 18], [18, 19], [19, 20]
    ]
    
    # 创建3D图
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 原始关键点
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(original_keypoints[:, 0], original_keypoints[:, 1], original_keypoints[:, 2], 
                c='blue', label='Original Keypoints', s=50)
    # 添加关键点标签
    for i, (x, y, z) in enumerate(original_keypoints):
        ax1.text(x, y, z, str(i), fontsize=8)
    # 绘制所有连接
    for connection in connections:
        start_idx, end_idx = connection
        ax1.plot([original_keypoints[start_idx, 0], original_keypoints[end_idx, 0]],
                 [original_keypoints[start_idx, 1], original_keypoints[end_idx, 1]],
                 [original_keypoints[start_idx, 2], original_keypoints[end_idx, 2]], 'g-', alpha=0.5)
    ax1.set_title('Original Keypoints')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    
    # 2. ICP转换后的关键点
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(transformed_keypoints[:, 0], transformed_keypoints[:, 1], transformed_keypoints[:, 2], 
                c='green', label='Transformed Keypoints', s=50)
    # 添加关键点标签
    for i, (x, y, z) in enumerate(transformed_keypoints):
        ax2.text(x, y, z, str(i), fontsize=8)
    # 绘制所有连接
    for connection in connections:
        start_idx, end_idx = connection
        ax2.plot([transformed_keypoints[start_idx, 0], transformed_keypoints[end_idx, 0]],
                 [transformed_keypoints[start_idx, 1], transformed_keypoints[end_idx, 1]],
                 [transformed_keypoints[start_idx, 2], transformed_keypoints[end_idx, 2]], 'g-', alpha=0.5)
    ax2.set_title('After ICP Transformation')
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    
    # 3. 关节约束后的关键点
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(constrained_keypoints[:, 0], constrained_keypoints[:, 1], constrained_keypoints[:, 2], 
                c='red', label='Constrained Keypoints', s=50)
    # 添加关键点标签
    for i, (x, y, z) in enumerate(constrained_keypoints):
        ax3.text(x, y, z, str(i), fontsize=8)
    # 绘制所有连接
    for connection in connections:
        start_idx, end_idx = connection
        ax3.plot([constrained_keypoints[start_idx, 0], constrained_keypoints[end_idx, 0]],
                 [constrained_keypoints[start_idx, 1], constrained_keypoints[end_idx, 1]],
                 [constrained_keypoints[start_idx, 2], constrained_keypoints[end_idx, 2]], 'g-', alpha=0.5)
    ax3.set_title('After Joint Constraints')
    ax3.set_xlabel('X axis')
    ax3.set_ylabel('Y axis')
    ax3.set_zlabel('Z axis')
    
    # 设置统一的视角（经典斜上方）
    for ax in [ax1, ax2, ax3]:
        ax.view_init(elev=20, azim=45)
        # 设置坐标轴样式
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    # 保存经典视角
    output_path = os.path.join(output_dir, "keypoints_transformation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"关键点变换可视化已保存至: {output_path}")
    # 额外保存matplotlib默认视角
    for ax in [ax1, ax2, ax3]:
        ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    output_path2 = os.path.join(output_dir, "keypoints_transformation_matplotlibview.png")
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"关键点变换matplotlib视角可视化已保存至: {output_path2}")
    plt.close()

def visualize_robot_action(keypoints_3d, pt, Rt, output_dir):
    """
    可视化机器人动作
    
    参数:
        keypoints_3d: numpy array, shape (21, 3), 最终的关键点
        pt: numpy array, shape (3,), 末端执行器位置
        Rt: numpy array, shape (6,), 末端执行器方向
        output_dir: str, 输出目录
    """
    # 定义手部关键点连接关系
    connections = [
        # 拇指
        [0, 1], [1, 2], [2, 3], [3, 4],
        # 食指
        [0, 5], [5, 6], [6, 7], [7, 8],
        # 中指
        [0, 9], [9, 10], [10, 11], [11, 12],
        # 无名指
        [0, 13], [13, 14], [14, 15], [15, 16],
        # 小指
        [0, 17], [17, 18], [18, 19], [19, 20]
    ]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制关键点
    ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2], 
               c='blue', label='Hand Keypoints', s=50)
    
    # 添加关键点标签
    for i, (x, y, z) in enumerate(keypoints_3d):
        ax.text(x, y, z, str(i), fontsize=8)
    
    # 绘制所有连接
    for connection in connections:
        start_idx, end_idx = connection
        ax.plot([keypoints_3d[start_idx, 0], keypoints_3d[end_idx, 0]],
                [keypoints_3d[start_idx, 1], keypoints_3d[end_idx, 1]],
                [keypoints_3d[start_idx, 2], keypoints_3d[end_idx, 2]], 'g-', alpha=0.5)
    
    # 绘制末端执行器位置
    ax.scatter(pt[0], pt[1], pt[2], c='red', s=100, label='End-effector Position')
    
    # 绘制方向向量
    normal = Rt[:3]  # 法向量
    principal_axis = Rt[3:]  # 主方向
    
    # 绘制法向量
    ax.quiver(pt[0], pt[1], pt[2], 
              normal[0], normal[1], normal[2],
              color='green', label='Normal Vector', length=0.1)
    
    # 绘制主方向
    ax.quiver(pt[0], pt[1], pt[2], 
              principal_axis[0], principal_axis[1], principal_axis[2],
              color='purple', label='Principal Axis', length=0.1)
    
    # 设置标题和标签
    ax.set_title('Robot Action Visualization')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    
    # 设置经典视角
    ax.view_init(elev=20, azim=45)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.grid(True, linestyle='--', alpha=0.3)
    # 保存经典视角
    output_path = os.path.join(output_dir, "robot_action.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"机器人动作可视化已保存至: {output_path}")
    # 额外保存matplotlib默认视角
    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    output_path2 = os.path.join(output_dir, "robot_action_matplotlibview.png")
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"机器人动作matplotlib视角可视化已保存至: {output_path2}")
    plt.close()

def visualize_paper_style(keypoints_3d, pt, Rt, output_dir):
    """
    以论文Fig.3风格可视化手部关键点和机器人动作参数
    
    参数:
        keypoints_3d: numpy array, shape (21, 3), 关键点
        pt: numpy array, shape (3,), 末端执行器位置
        Rt: numpy array, shape (6,), 末端执行器方向
        output_dir: str, 输出目录
    """
    fig = plt.figure(figsize=(16, 8))
    
    # 创建两个子图，不同视角
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 为两个视角绘制相同的内容
    for ax_idx, ax in enumerate([ax1, ax2]):
        # 定义手部骨架连接关系
        connections = [
            # 拇指
            [0, 1], [1, 2], [2, 3], [3, 4],
            # 食指
            [0, 5], [5, 6], [6, 7], [7, 8],
            # 中指
            [0, 9], [9, 10], [10, 11], [11, 12],
            # 无名指
            [0, 13], [13, 14], [14, 15], [15, 16],
            # 小指
            [0, 17], [17, 18], [18, 19], [19, 20]
        ]
        
        # 获取拇指和食指关键点
        thumb_indices = [1, 2, 3, 4]  # 拇指关键点
        index_indices = [5, 6, 7, 8]  # 食指关键点
        
        thumb_points = keypoints_3d[thumb_indices]
        index_points = keypoints_3d[index_indices]
        
        # 合并拇指和食指点用于平面拟合
        all_points = np.vstack([thumb_points, index_points])
        
        # 法向量（从Rt获取）
        normal = Rt[:3]  # 使用传入的法向量，确保与动作计算一致
        
        # 计算平面上的点（用于可视化平面）
        center = np.mean(all_points, axis=0)
        
        # 生成平面点
        xx, yy = np.meshgrid(np.linspace(min(all_points[:,0])-0.02, max(all_points[:,0])+0.02, 10),
                             np.linspace(min(all_points[:,1])-0.02, max(all_points[:,1])+0.02, 10))
        
        # 计算平面上对应的z值 (ax + by + cz + d = 0 -> z = (-ax - by - d) / c)
        d = -np.dot(normal, center)
        zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2] if abs(normal[2]) > 1e-6 else np.zeros_like(xx)
        
        # 绘制半透明抓取平面
        surf = ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray', label='Grasp Plane')
        
        # 先绘制整个手部骨架（黑色线条）
        for connection in connections:
            start_idx, end_idx = connection
            # 拇指和食指用黑色细线，后面会单独高亮
            line_color = 'black'
            line_width = 1.0
            ax.plot([keypoints_3d[start_idx, 0], keypoints_3d[end_idx, 0]],
                    [keypoints_3d[start_idx, 1], keypoints_3d[end_idx, 1]],
                    [keypoints_3d[start_idx, 2], keypoints_3d[end_idx, 2]], 
                    color=line_color, linewidth=line_width)
        
        # 绘制拇指关键点（橙色）
        ax.scatter(thumb_points[:, 0], thumb_points[:, 1], thumb_points[:, 2], 
                  c='orange', s=50, label='Thumb')
        
        # 绘制食指关键点（蓝色）
        ax.scatter(index_points[:, 0], index_points[:, 1], index_points[:, 2], 
                  c='blue', s=50, label='Index')
        
        # 特别标记拇指尖和食指尖
        ax.scatter(thumb_points[-1, 0], thumb_points[-1, 1], thumb_points[-1, 2], 
                  c='darkorange', s=100, edgecolor='black', label='Thumb Tip')
        ax.scatter(index_points[-1, 0], index_points[-1, 1], index_points[-1, 2], 
                  c='royalblue', s=100, edgecolor='black', label='Index Tip')
        
        # 强调拇指和食指的连接线
        # 拇指连接
        for i in range(len(thumb_points) - 1):
            ax.plot([keypoints_3d[i+1, 0], keypoints_3d[i+2, 0]],
                    [keypoints_3d[i+1, 1], keypoints_3d[i+2, 1]],
                    [keypoints_3d[i+1, 2], keypoints_3d[i+2, 2]], 
                    'orange', linewidth=2)
        
        # 食指连接
        for i in range(len(index_points) - 1):
            ax.plot([keypoints_3d[i+5, 0], keypoints_3d[i+6, 0]],
                    [keypoints_3d[i+5, 1], keypoints_3d[i+6, 1]],
                    [keypoints_3d[i+5, 2], keypoints_3d[i+6, 2]], 
                    'blue', linewidth=2)
        
        # 添加拇指和食指的文本标注
        ax.text(thumb_points[1, 0], thumb_points[1, 1], thumb_points[1, 2], 
                "$x_t^{thumb}$", color='orange', fontsize=10)
        ax.text(index_points[1, 0], index_points[1, 1], index_points[1, 2], 
                "$x_t^{index}$", color='blue', fontsize=10)
        
        # 添加指尖文本标注
        ax.text(thumb_points[-1, 0], thumb_points[-1, 1], thumb_points[-1, 2], 
                "$x_t^{thumb,tip}$", color='orange', fontsize=10)
        ax.text(index_points[-1, 0], index_points[-1, 1], index_points[-1, 2], 
                "$x_t^{index,tip}$", color='blue', fontsize=10)
        
        # 绘制目标位置pt（拇指尖和食指尖的中点）
        ax.scatter(pt[0], pt[1], pt[2], c='red', s=150, marker='o')
        ax.text(pt[0], pt[1], pt[2], "$p_t$", color='red', fontsize=12)
        
        # 添加指尖到目标位置的虚线
        ax.plot([thumb_points[-1, 0], pt[0]], 
                [thumb_points[-1, 1], pt[1]], 
                [thumb_points[-1, 2], pt[2]], 'gray', linestyle='--', linewidth=1)
        ax.plot([index_points[-1, 0], pt[0]], 
                [index_points[-1, 1], pt[1]], 
                [index_points[-1, 2], pt[2]], 'gray', linestyle='--', linewidth=1)
        
        # 绘制坐标系表示方向Rt
        arrow_length = 0.02
        normal_vec = Rt[:3]  # 法向量
        principal_axis = Rt[3:]  # 主轴
        
        # 确保主轴与法向量垂直
        principal_axis = principal_axis - np.dot(principal_axis, normal_vec) * normal_vec
        if np.linalg.norm(principal_axis) > 1e-8:
            principal_axis = principal_axis / np.linalg.norm(principal_axis)
        
        # 计算第三个轴（垂直于前两个）
        third_axis = np.cross(normal_vec, principal_axis)
        if np.linalg.norm(third_axis) > 1e-8:
            third_axis = third_axis / np.linalg.norm(third_axis)
        
        # 绘制三个方向轴
        ax.quiver(pt[0], pt[1], pt[2], 
                  normal_vec[0], normal_vec[1], normal_vec[2],
                  color='red', length=arrow_length, linewidth=2)
        ax.quiver(pt[0], pt[1], pt[2], 
                  principal_axis[0], principal_axis[1], principal_axis[2],
                  color='blue', length=arrow_length, linewidth=2)
        ax.quiver(pt[0], pt[1], pt[2], 
                  third_axis[0], third_axis[1], third_axis[2],
                  color='green', length=arrow_length, linewidth=2)
        
        # 添加箭头标签解释
        offset = 0.01  # 标签偏移量
        ax.text(pt[0] + normal_vec[0]*arrow_length + offset, 
                pt[1] + normal_vec[1]*arrow_length + offset, 
                pt[2] + normal_vec[2]*arrow_length + offset, 
                "Normal (Plane)", color='red', fontsize=8)
        ax.text(pt[0] + principal_axis[0]*arrow_length + offset, 
                pt[1] + principal_axis[1]*arrow_length + offset, 
                pt[2] + principal_axis[2]*arrow_length + offset, 
                "Finger Dir.", color='blue', fontsize=8)
        ax.text(pt[0] + third_axis[0]*arrow_length + offset, 
                pt[1] + third_axis[1]*arrow_length + offset, 
                pt[2] + third_axis[2]*arrow_length + offset, 
                "Third Axis", color='green', fontsize=8)
        
        # 添加"Grasp Plane"文字标签
        text_pos = np.mean(all_points, axis=0) + normal_vec * 0.02
        ax.text(text_pos[0], text_pos[1], text_pos[2], "Grasp Plane", 
                color='black', fontsize=10, ha='center')
        
        # 设置坐标轴和视角
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 关闭坐标轴刻度，使图形更接近论文样式
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # 调整坐标轴范围使图形居中
        x_range = max(keypoints_3d[:,0]) - min(keypoints_3d[:,0])
        y_range = max(keypoints_3d[:,1]) - min(keypoints_3d[:,1])
        z_range = max(keypoints_3d[:,2]) - min(keypoints_3d[:,2])
        max_range = max(x_range, y_range, z_range) * 0.6
        
        mid_x = (max(keypoints_3d[:,0]) + min(keypoints_3d[:,0])) / 2
        mid_y = (max(keypoints_3d[:,1]) + min(keypoints_3d[:,1])) / 2
        mid_z = (max(keypoints_3d[:,2]) + min(keypoints_3d[:,2])) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置不同的视角
    # 第一个视角：类似论文中的斜视角，可以看到手部结构
    ax1.view_init(elev=20, azim=45)
    ax1.set_title("Side View", fontsize=12)
    
    # 第二个视角：正对拇指和食指之间的抓取区域
    # 计算拇指根部到食指根部的向量（作为"左右"方向）
    thumb_base = keypoints_3d[1]  # 拇指根部
    index_base = keypoints_3d[5]  # 食指根部
    side_vec = index_base - thumb_base
    side_vec = side_vec / np.linalg.norm(side_vec)
    
    # 计算从手掌中心到抓取点的向量（作为"前后"方向）
    palm_center = keypoints_3d[0]  # 手掌中心
    forward_vec = pt - palm_center
    forward_vec = forward_vec / np.linalg.norm(forward_vec)
    
    # 计算垂直于这两个向量的方向（作为"上下"方向）
    up_vec = np.cross(side_vec, forward_vec)
    up_vec = up_vec / np.linalg.norm(up_vec)
    
    # 精确计算视角，让观察者正对拇指和食指之间
    # 我们希望视线方向与side_vec平行但反向，这样可以看到拇指和食指夹角
    view_vec = -side_vec
    
    # 将视图向量转换为球坐标系的方位角和仰角
    azim = np.degrees(np.arctan2(view_vec[1], view_vec[0]))
    elev = np.degrees(np.arcsin(view_vec[2]))
    
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_title("Front View (Between Thumb and Index)", fontsize=14)
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, "paper_style_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"论文风格可视化已保存至: {output_path}")
    
    # 同时保存单独的正视图（正对抓取平面）
    fig_front = plt.figure(figsize=(8, 8))
    ax_front = fig_front.add_subplot(111, projection='3d')
    
    # 复制之前所有绘图元素
    # 先绘制整个手部骨架（黑色线条）
    for connection in connections:
        start_idx, end_idx = connection
        line_color = 'black'
        line_width = 1.0
        ax_front.plot([keypoints_3d[start_idx, 0], keypoints_3d[end_idx, 0]],
                [keypoints_3d[start_idx, 1], keypoints_3d[end_idx, 1]],
                [keypoints_3d[start_idx, 2], keypoints_3d[end_idx, 2]], 
                color=line_color, linewidth=line_width)
    
    # 绘制关键点和连接线
    ax_front.scatter(thumb_points[:, 0], thumb_points[:, 1], thumb_points[:, 2], c='orange', s=50)
    ax_front.scatter(index_points[:, 0], index_points[:, 1], index_points[:, 2], c='blue', s=50)
    ax_front.scatter(thumb_points[-1, 0], thumb_points[-1, 1], thumb_points[-1, 2], c='darkorange', s=100, edgecolor='black')
    ax_front.scatter(index_points[-1, 0], index_points[-1, 1], index_points[-1, 2], c='royalblue', s=100, edgecolor='black')
    
    # 强调拇指和食指连接
    for i in range(len(thumb_points) - 1):
        ax_front.plot([keypoints_3d[i+1, 0], keypoints_3d[i+2, 0]],
                [keypoints_3d[i+1, 1], keypoints_3d[i+2, 1]],
                [keypoints_3d[i+1, 2], keypoints_3d[i+2, 2]], 'orange', linewidth=2)
    
    for i in range(len(index_points) - 1):
        ax_front.plot([keypoints_3d[i+5, 0], keypoints_3d[i+6, 0]],
                [keypoints_3d[i+5, 1], keypoints_3d[i+6, 1]],
                [keypoints_3d[i+5, 2], keypoints_3d[i+6, 2]], 'blue', linewidth=2)
    
    # 绘制平面
    surf = ax_front.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    # 添加目标位置和坐标轴
    ax_front.scatter(pt[0], pt[1], pt[2], c='red', s=150, marker='o')
    ax_front.quiver(pt[0], pt[1], pt[2], normal_vec[0], normal_vec[1], normal_vec[2], color='red', length=arrow_length, linewidth=2)
    ax_front.quiver(pt[0], pt[1], pt[2], principal_axis[0], principal_axis[1], principal_axis[2], color='blue', length=arrow_length, linewidth=2)
    ax_front.quiver(pt[0], pt[1], pt[2], third_axis[0], third_axis[1], third_axis[2], color='green', length=arrow_length, linewidth=2)
    
    # 添加箭头标签解释
    offset = 0.01  # 标签偏移量
    ax_front.text(pt[0] + normal_vec[0]*arrow_length + offset, 
            pt[1] + normal_vec[1]*arrow_length + offset, 
            pt[2] + normal_vec[2]*arrow_length + offset, 
            "Normal (Plane)", color='red', fontsize=8)
    ax_front.text(pt[0] + principal_axis[0]*arrow_length + offset, 
            pt[1] + principal_axis[1]*arrow_length + offset, 
            pt[2] + principal_axis[2]*arrow_length + offset, 
            "Finger Dir.", color='blue', fontsize=8)
    ax_front.text(pt[0] + third_axis[0]*arrow_length + offset, 
            pt[1] + third_axis[1]*arrow_length + offset, 
            pt[2] + third_axis[2]*arrow_length + offset, 
            "Third Axis", color='green', fontsize=8)
    
    # 设置标题和视角
    ax_front.set_title("Front View (Between Thumb and Index)", fontsize=14)
    ax_front.set_xticks([])
    ax_front.set_yticks([])
    ax_front.set_zticks([])
    ax_front.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_front.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_front.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置为正对平面视角
    ax_front.view_init(elev=elev, azim=azim)
    
    front_view_path = os.path.join(output_dir, "thumb_index_view.png")
    plt.savefig(front_view_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"拇指和食指之间视图已保存至: {front_view_path}")
    
    plt.close('all')

def main():
    """
    处理一组视频帧的HaMeR输出，应用ICP转换，生成机器人动作
    """
    # 基础目录
    hamer_base_dir = "/root/autodl-tmp/phantom_data/EgoDexter/EgoDexter/data/Desk/hamer_output"
    keypoints_dir = os.path.join(hamer_base_dir, "keypoints_3d")
    icp_results_dir = os.path.join(hamer_base_dir, "icp_results")
    robot_actions_dir = os.path.join(hamer_base_dir, "robot_actions")
    
    # 创建输出目录
    os.makedirs(robot_actions_dir, exist_ok=True)
    
    # 获取ICP结果文件列表
    icp_files = [f for f in os.listdir(icp_results_dir) if f.endswith('_icp_registration_info.json')]
    
    print(f"找到 {len(icp_files)} 个ICP配准结果文件")
    processed_count = 0
    error_count = 0
    
    for icp_file in icp_files:
        try:
            # 提取图像名称和手的ID
            # 格式: image_XXXXX_color_handY_icp_registration_info.json
            parts = icp_file.split('_')
            image_name = f"{parts[0]}_{parts[1]}_{parts[2]}"
            hand_id = parts[3].replace('hand', '')
            
            # 构建HaMeR关键点文件路径
            keypoints_file = f"{image_name}_{hand_id}_keypoints.json"
            keypoints_path = os.path.join(keypoints_dir, keypoints_file)
            
            # 检查关键点文件是否存在
            if not os.path.exists(keypoints_path):
                print(f"警告: 找不到关键点文件: {keypoints_path}")
                continue
            
            # 构建ICP信息文件完整路径
            icp_info_path = os.path.join(icp_results_dir, icp_file)
            
            # 处理HaMeR输出和ICP信息
        action = process_hamer_output(keypoints_path, icp_info_path)
            
            # 构建输出文件路径
            robot_action_file = f"{image_name}_{hand_id}_robot_action.json"
            output_path = os.path.join(robot_actions_dir, robot_action_file)
        
        # 保存动作数据
        with open(output_path, 'w') as f:
            json.dump(action, f, indent=2)
            
            # 创建帧特定的可视化目录
            frame_viz_dir = os.path.join(robot_actions_dir, f"{image_name}_{hand_id}_viz")
            os.makedirs(frame_viz_dir, exist_ok=True)
        
        # 可视化关键点变换过程
        visualize_keypoints(
            np.array(action['original_keypoints']),
            np.array(action['transformed_keypoints']),
            np.array(action['constrained_keypoints']),
                frame_viz_dir
        )
        
        # 可视化机器人动作
        visualize_robot_action(
            np.array(action['constrained_keypoints']),
            np.array(action['position']),
            np.array(action['orientation']),
                frame_viz_dir
        )
        
        # 添加论文风格可视化
        visualize_paper_style(
            np.array(action['constrained_keypoints']),
            np.array(action['position']),
            np.array(action['orientation']),
                frame_viz_dir
            )
            
            processed_count += 1
            print(f"已处理 ({processed_count}/{len(icp_files)}): {image_name}_{hand_id}")
            
            # 每处理10个文件输出一次进度
            if processed_count % 10 == 0:
                print(f"进度: {processed_count}/{len(icp_files)} ({processed_count/len(icp_files)*100:.1f}%)")
        
        except Exception as e:
            error_count += 1
            print(f"处理文件 {icp_file} 时出错: {str(e)}")
    
    print(f"\n处理完成! 成功处理 {processed_count}/{len(icp_files)} 个文件, {error_count} 个错误")
    if processed_count > 0:
        print(f"机器人动作输出目录: {robot_actions_dir}")
        print(f"每帧的可视化结果位于对应帧的 *_viz 子目录中")

if __name__ == "__main__":
    main() 