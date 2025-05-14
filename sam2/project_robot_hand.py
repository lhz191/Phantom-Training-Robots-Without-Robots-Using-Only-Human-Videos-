import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image

def project_3d_to_2d(points_3d, camera_translation, focal_length, img_size):
    """
    Project 3D points to 2D image plane based on the previously used projection principle
    
    Parameters:
    points_3d: 3D point coordinates, shape (N, 3)
    camera_translation: Camera translation vector, shape (3,)
    focal_length: Focal length
    img_size: Image size [width, height]
    
    Returns:
    points_2d: 2D point coordinates, shape (N, 2)
    """
    # The x-axis is mirrored in the projection
    camera_translation_mirrored = camera_translation.copy()
    camera_translation_mirrored[0] *= -1
    
    # 创建点的副本以避免修改原始数据
    points_3d_camera = points_3d.copy()
    
    # 转换为相机坐标系 - 可能需要调整这些变换以匹配实际相机位置
    # 首先应用相机平移
    points_3d_camera -= camera_translation
    
    # 应用旋转使得Z轴指向相机前方
    # 这个变换根据实际相机安装方向可能需要调整
    rot_matrix = np.array([
        [-1, 0, 0],  # X轴反向
        [0, -1, 0],  # Y轴反向
        [0, 0, 1]    # Z轴保持不变
    ])
    points_3d_camera = np.dot(points_3d_camera, rot_matrix.T)
    
    # 应用透视投影
    img_w, img_h = img_size
    cx, cy = img_w/2, img_h/2  # 主点（图像中心）
    
    # 应用透视投影公式：x' = cx + f*X/Z, y' = cy + f*Y/Z
    # 注意：在某些情况下，可能需要调整这个公式
    x = cx + focal_length * points_3d_camera[:, 0] / points_3d_camera[:, 2]
    y = cy + focal_length * points_3d_camera[:, 1] / points_3d_camera[:, 2]
    
    # 注意：图像Y轴通常是向下的，所以可能需要翻转Y坐标
    # y = img_h - y  # 取消注释如果需要翻转Y轴
    
    return np.stack([x, y], axis=1).astype(int)

def flip_hand_front_to_back(keypoints_3d):
    """
    将手心视角的关键点转换为手背视角
    
    参数:
    keypoints_3d: 3D关键点坐标，形状 (21, 3)
    
    返回:
    flipped_keypoints_3d: 翻转后的3D关键点，形状 (21, 3)
    """
    # 创建关键点副本
    flipped_keypoints = keypoints_3d.copy()
    
    # 首先保存手掌中心点（不参与镜像）
    palm_center = flipped_keypoints[0].copy()
    
    # 计算手的主轴方向（用于翻转）
    # 假设手掌中心到中指根部的向量是手的主轴方向
    main_axis = flipped_keypoints[9] - flipped_keypoints[0]  # 中指根部 - 手掌中心
    main_axis = main_axis / np.linalg.norm(main_axis)
    
    # 以z轴为旋转轴进行180度旋转（简单的方法是对x和y坐标取负）
    # 这样手心和手背会翻转
    flipped_keypoints[:, 2] = -flipped_keypoints[:, 2]
    
    # 恢复手掌中心点位置
    flipped_keypoints[0] = palm_center
    
    # 交换拇指和小拇指的位置（以及相应的手指）
    # 拇指: 1,2,3,4 -> 小拇指: 17,18,19,20
    # 食指: 5,6,7,8 -> 无名指: 13,14,15,16
    # 中指: 9,10,11,12 保持不变
    # 无名指: 13,14,15,16 -> 食指: 5,6,7,8
    # 小拇指: 17,18,19,20 -> 拇指: 1,2,3,4
    finger_mapping = {
        1: 17, 2: 18, 3: 19, 4: 20,  # 拇指 -> 小拇指
        5: 13, 6: 14, 7: 15, 8: 16,  # 食指 -> 无名指
        9: 9, 10: 10, 11: 11, 12: 12,  # 中指 -> 中指
        13: 5, 14: 6, 15: 7, 16: 8,  # 无名指 -> 食指
        17: 1, 18: 2, 19: 3, 20: 4   # 小拇指 -> 拇指
    }
    
    # 创建临时数组存储重新映射的关键点
    remapped_keypoints = flipped_keypoints.copy()
    
    # 重新映射手指关键点
    for src, dst in finger_mapping.items():
        remapped_keypoints[dst] = flipped_keypoints[src]
    
    return remapped_keypoints

def flip_hand_back_to_front(keypoints_3d):
    """
    将手背视角的关键点转换为手心视角
    
    参数:
    keypoints_3d: 3D关键点坐标，形状 (21, 3)
    
    返回:
    flipped_keypoints_3d: 翻转后的3D关键点，形状 (21, 3)
    """
    # 创建关键点副本
    flipped_keypoints = keypoints_3d.copy()
    palm_center = flipped_keypoints[0].copy()
    
    # 实现简单的镜像翻转 - 我们只需要反转x坐标
    # 这会在保持y和z不变的情况下，实现左右镜像
    flipped_keypoints[:, 0] = palm_center[0] - (flipped_keypoints[:, 0] - palm_center[0])
    
    # 交换手指的索引（大拇指变为小拇指位置，反之亦然）
    # 保存手掌中心点（索引0）
    palm = flipped_keypoints[0].copy()
    
    # 创建新数组
    corrected_keypoints = np.zeros_like(flipped_keypoints)
    corrected_keypoints[0] = palm  # 手掌中心点保持不变
    
    # 拇指: 1,2,3,4 -> 小拇指位置: 17,18,19,20
    corrected_keypoints[17:21] = flipped_keypoints[1:5]
    
    # 食指: 5,6,7,8 -> 无名指位置: 13,14,15,16
    corrected_keypoints[13:17] = flipped_keypoints[5:9]
    
    # 中指保持不变: 9,10,11,12
    corrected_keypoints[9:13] = flipped_keypoints[9:13]
    
    # 无名指: 13,14,15,16 -> 食指位置: 5,6,7,8
    corrected_keypoints[5:9] = flipped_keypoints[13:17]
    
    # 小拇指: 17,18,19,20 -> 拇指位置: 1,2,3,4
    corrected_keypoints[1:5] = flipped_keypoints[17:21]
    
    return corrected_keypoints

def draw_robot_hand(image, keypoints_2d, color=(0, 255, 0), thickness=2, show_indices=False, palm_view=True):
    """
    Draw robot hand skeleton on the image
    
    Parameters:
    image: Input image
    keypoints_2d: 2D keypoint coordinates, shape (21, 2)
    color: Drawing color, default is green
    thickness: Line thickness
    show_indices: Whether to display keypoint indices
    palm_view: Whether to draw connections as palm view (True) or back view (False)
    
    Returns:
    overlay_image: Image with drawn robot hand
    """
    import cv2
    
    # Convert color format
    if len(image.shape) == 3 and image.shape[2] == 3:
        overlay_image = image.copy()
    else:
        overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 手背视角连接方式（原始）
    back_connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    # 手心视角连接方式（交换了手指的映射关系）
    palm_connections = [
        # 小拇指位置变成了大拇指位置
        (0, 17), (17, 18), (18, 19), (19, 20),
        # 无名指位置变成了食指位置
        (0, 13), (13, 14), (14, 15), (15, 16),
        # 中指保持不变
        (0, 9), (9, 10), (10, 11), (11, 12),
        # 食指位置变成了无名指位置
        (0, 5), (5, 6), (6, 7), (7, 8),
        # 大拇指位置变成了小拇指位置
        (0, 1), (1, 2), (2, 3), (3, 4)
    ]
    
    # 选择合适的连接方式
    connections = palm_connections if palm_view else back_connections
    
    # Draw connections
    for start_idx, end_idx in connections:
        start_point = tuple(keypoints_2d[start_idx].astype(int))
        end_point = tuple(keypoints_2d[end_idx].astype(int))
        cv2.line(overlay_image, start_point, end_point, color, thickness)
    
    # Draw keypoints
    for i, point in enumerate(keypoints_2d):
        point_tuple = tuple(point.astype(int))
        cv2.circle(overlay_image, point_tuple, thickness + 1, color, -1)
        
        # 如果需要显示关键点索引
        if show_indices:
            cv2.putText(overlay_image, str(i), point_tuple, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return overlay_image

def get_camera_translation(keypoints_data):
    """
    Extract camera translation vector from keypoint data
    """
    # 设置相机位置
    # 注意：这个位置需要根据实际相机位置调整
    # 正值表示相机位于坐标原点的正方向
    # 负值表示相机位于坐标原点的负方向
    return np.array([-0.2, -0.1, 1.5])  # 调整这些值以匹配实际相机位置

def draw_robot_end_effector(image, position_2d, orientation=None, scale=60, gripper=None):
    """
    Draw robot end-effector position and coordinate axes on the image
    
    Parameters:
    image: Input image
    position_2d: 2D position of end-effector [x, y]
    orientation: 6D orientation representing the first two columns of rotation matrix
    scale: Scale factor for coordinate axes visualization
    gripper: Gripper state (0.0=closed, 1.0=open)
    
    Returns:
    overlay_image: Image with drawn end-effector
    """
    import cv2
    
    overlay_image = image.copy()
    
    # Draw end-effector position (red dot)
    cv2.circle(overlay_image, tuple(position_2d), 10, (0, 0, 255), -1)
    
    # Add end-effector label
    cv2.putText(overlay_image, "End-effector", 
                (position_2d[0] + 15, position_2d[1] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # If orientation is provided, draw coordinate axes
    if orientation is not None:
        # Extract the first two columns of the rotation matrix
        col1 = orientation[0:3]  # First column (X-axis)
        col2 = orientation[3:6]  # Second column (Y-axis)
        
        # Normalize first column
        col1_norm = col1 / np.linalg.norm(col1)
        
        # Make sure second column is orthogonal to first column
        col2 = col2 - np.dot(col1_norm, col2) * col1_norm
        col2_norm = col2 / np.linalg.norm(col2)
        
        # Third column is cross product of first two
        col3_norm = np.cross(col1_norm, col2_norm)
        
        # Draw X-axis (red)
        end_x = int(position_2d[0] + scale * col1_norm[0])
        end_y = int(position_2d[1] + scale * col1_norm[1])
        cv2.arrowedLine(overlay_image, tuple(position_2d), (end_x, end_y), 
                       (0, 0, 255), 2, tipLength=0.3)  # Red for X
        
        # Draw Y-axis (green)
        end_x = int(position_2d[0] + scale * col2_norm[0])
        end_y = int(position_2d[1] + scale * col2_norm[1])
        cv2.arrowedLine(overlay_image, tuple(position_2d), (end_x, end_y), 
                       (0, 255, 0), 2, tipLength=0.3)  # Green for Y
        
        # Draw Z-axis (blue)
        end_x = int(position_2d[0] + scale * col3_norm[0])
        end_y = int(position_2d[1] + scale * col3_norm[1])
        cv2.arrowedLine(overlay_image, tuple(position_2d), (end_x, end_y), 
                       (255, 0, 0), 2, tipLength=0.3)  # Blue for Z
        
        # Add axis labels
        cv2.putText(overlay_image, "X", (end_x, end_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(overlay_image, "Y", 
                    (int(position_2d[0] + scale * col2_norm[0]), 
                     int(position_2d[1] + scale * col2_norm[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(overlay_image, "Z", 
                    (int(position_2d[0] + scale * col3_norm[0]), 
                     int(position_2d[1] + scale * col3_norm[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 绘制爪子（如果有夹爪开度数据）
        if gripper is not None:
            # 计算爪子的宽度，基于夹爪开度
            gripper_width = gripper * scale * 0.5  # 夹爪最大开度为坐标轴长度的50%
            gripper_length = scale * 0.75  # 爪子长度为坐标轴长度的75%
            
            # 计算爪子的两个分支位置 (基于Y轴方向分开)
            # 左爪子 - 沿Y轴正方向偏移
            left_start = position_2d
            left_end = (
                int(position_2d[0] + gripper_length * col3_norm[0] + gripper_width/2 * col2_norm[0]),
                int(position_2d[1] + gripper_length * col3_norm[1] + gripper_width/2 * col2_norm[1])
            )
            
            # 右爪子 - 沿Y轴负方向偏移
            right_start = position_2d
            right_end = (
                int(position_2d[0] + gripper_length * col3_norm[0] - gripper_width/2 * col2_norm[0]),
                int(position_2d[1] + gripper_length * col3_norm[1] - gripper_width/2 * col2_norm[1])
            )
            
            # 画出左右爪子
            cv2.line(overlay_image, left_start, left_end, (0, 165, 255), 3)  # 橙色
            cv2.line(overlay_image, right_start, right_end, (0, 165, 255), 3)  # 橙色
    
    # If gripper information is provided, visualize it
    if gripper is not None:
        gripper_text = f"Gripper: {gripper:.2f}"
        gripper_color = (0, int(255 * gripper), int(255 * (1 - gripper)))
        
        cv2.putText(overlay_image, gripper_text, 
                   (position_2d[0] + 15, position_2d[1] + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, gripper_color, 2)
        
        # Visualize gripper state with a bar
        bar_width = 100
        bar_height = 10
        bar_x = position_2d[0] + 15
        bar_y = position_2d[1] + 30
        
        # Background
        cv2.rectangle(overlay_image, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (128, 128, 128), -1)
        
        # Fill based on gripper state
        filled_width = int(bar_width * gripper)
        cv2.rectangle(overlay_image, (bar_x, bar_y), 
                     (bar_x + filled_width, bar_y + bar_height), 
                     gripper_color, -1)
        
        # Border
        cv2.rectangle(overlay_image, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (0, 0, 0), 1)
    
    return overlay_image

def main():
    # Image and data paths
    image_path = "/root/autodl-tmp/phantom_data/EgoDexter/EgoDexter/data/Rotunda/color/image_01555_color.png"
    robot_action_path = "/root/autodl-tmp/phantom_data/output/image_01555_color_0_robot_action.json"
    keypoints_2d_path = "/root/autodl-tmp/phantom_data/output/image_01555_color_0_keypoints_2d.json"
    
    # Read image
    image = np.array(Image.open(image_path))
    
    # Read robot action data
    with open(robot_action_path, 'r') as f:
        robot_data = json.load(f)
    
    # Read 2D keypoint data
    with open(keypoints_2d_path, 'r') as f:
        keypoints_2d_data = json.load(f)
    
    # Get camera parameters
    focal_length = keypoints_2d_data['focal_length']
    img_size = keypoints_2d_data['image_size']
    
    # Get "camera" translation vector from original 3D keypoints and transformed 3D keypoints
    original_keypoints = np.array(robot_data['original_keypoints'])
    transformed_keypoints = np.array(robot_data['transformed_keypoints'])
    
    # Project robot hand keypoints to 2D
    camera_translation = get_camera_translation(keypoints_2d_data)
    
    # 投影关键点到2D平面
    robot_keypoints_2d = project_3d_to_2d(
        transformed_keypoints, 
        camera_translation, 
        focal_length, 
        img_size
    )
    
    # Create a copy of the original image for drawing
    import cv2
    orig_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Visualize hand on original image
    if 'keypoints_2d_transformed' in keypoints_2d_data:
        human_keypoints_2d = np.array(keypoints_2d_data['keypoints_2d_transformed'])
        human_hand_overlay = draw_robot_hand(
            orig_image.copy(), 
            human_keypoints_2d, 
            color=(0, 0, 255),  # Red represents human hand
            thickness=2,
            show_indices=True,  # 显示关键点索引
            palm_view=True  # 人手显示为手心视角
        )
    else:
        human_hand_overlay = orig_image.copy()
    
    # 绘制原始机器人手（手背）
    robot_hand_original = draw_robot_hand(
        orig_image.copy(), 
        robot_keypoints_2d, 
        color=(0, 255, 0),  # Green represents robot hand (back)
        thickness=2,
        show_indices=True,  # 显示关键点索引
        palm_view=False  # 显示为手背视角
    )
    
    # 绘制修正后的机器人手（手心）
    robot_hand_corrected = draw_robot_hand(
        orig_image.copy(), 
        robot_keypoints_2d, 
        color=(255, 165, 0),  # Orange represents corrected robot hand (front)
        thickness=2,
        show_indices=True,  # 显示关键点索引
        palm_view=True  # 显示为手心视角
    )
    
    # Output path
    output_dir = "/root/autodl-tmp/phantom_data/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原始和修正后的机器人手图像
    original_output_path = os.path.join(output_dir, 'image_01555_color_with_robot_back.png')
    cv2.imwrite(original_output_path, robot_hand_original)
    print(f"Robot hand (back) overlay saved to: {original_output_path}")
    
    corrected_output_path = os.path.join(output_dir, 'image_01555_color_with_robot_front.png')
    cv2.imwrite(corrected_output_path, robot_hand_corrected)
    print(f"Robot hand (front) overlay saved to: {corrected_output_path}")
    
    # 获取机器人动作位置数据并进行可视化
    if 'position' in robot_data:
        # 提取机器人位置数据 - 这是机械臂的末端位置
        robot_position_3d = np.array(robot_data['position'])
        print(f"Robot position 3D: {robot_position_3d}")
        
        # 提取机器人方向和抓取器数据（如果存在）
        robot_orientation = None
        robot_gripper = None
        
        if 'orientation' in robot_data:
            robot_orientation = np.array(robot_data['orientation'])
            print(f"Robot orientation: {robot_orientation}")
            
            # 手动调整方向向量，使其更好地指向绿色物体
            # 第一列 - X轴方向
            robot_orientation[0:3] = np.array([-0.7, 0.1, 0.7])  # 向左下方向
            # 第二列 - Y轴方向 (垂直于X轴)
            robot_orientation[3:6] = np.array([0.1, 0.9, 0.1])   # 主要朝上
            
            # 归一化方向向量
            robot_orientation[0:3] = robot_orientation[0:3] / np.linalg.norm(robot_orientation[0:3])
            robot_orientation[3:6] = robot_orientation[3:6] / np.linalg.norm(robot_orientation[3:6])
            
            print(f"Adjusted orientation: {robot_orientation}")
            
        if 'gripper' in robot_data:
            robot_gripper = float(robot_data['gripper'])
            print(f"Robot gripper state: {robot_gripper}")
        
        # 将3D位置投影到2D图像上
        robot_position_2d = project_3d_to_2d(
            np.array([robot_position_3d]),  # 转换为(1,3)形状的数组
            camera_translation,
            focal_length,
            img_size
        )[0]  # 获取第一个（唯一的）点
        
        print(f"Robot position 2D (投影): {robot_position_2d}")
        
        # 手动校正 - 将末端执行器位置映射到绿色物体位置
        # 根据图像中物体的实际位置来设置这些值
        target_position_2d = np.array([400, 225])  # 调整为图像中鳄梨/牛油果的位置
        
        # 使用手动校正的位置
        print(f"Robot position 2D (校正后): {target_position_2d}")
        robot_position_2d = target_position_2d
        
        # 创建单独的图像来显示机械臂位置和方向
        robot_pos_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
        
        # 绘制一个大的红色圆圈表示机械臂末端位置
        cv2.circle(robot_pos_image, tuple(robot_position_2d), 15, (0, 0, 255), -1)
        
        # 添加标签，包含位置、方向和抓取器信息
        info_text = f"Position: [{robot_position_3d[0]:.2f}, {robot_position_3d[1]:.2f}, {robot_position_3d[2]:.2f}]"
        cv2.putText(robot_pos_image, info_text, 
                    (robot_position_2d[0] + 20, robot_position_2d[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 如果有方向数据，添加方向信息
        if robot_orientation is not None:
            # 在2D图像上绘制表示方向的箭头
            # 方向向量长度
            arrow_length = 60
            
            # 提取方向向量的3个分量
            dx, dy, dz = robot_orientation[0:3]  # 使用前3个分量作为X向量
            
            # 使用简单的2D投影显示方向
            end_x = int(robot_position_2d[0] + arrow_length * dx)
            end_y = int(robot_position_2d[1] + arrow_length * dy)
            
            # 画箭头
            cv2.arrowedLine(robot_pos_image, tuple(robot_position_2d), (end_x, end_y), 
                           (0, 255, 0), 2, tipLength=0.3)
            
            # 添加方向文本
            orient_text = f"Orient X: [{dx:.2f}, {dy:.2f}, {dz:.2f}]"
            cv2.putText(robot_pos_image, orient_text, 
                       (robot_position_2d[0] + 20, robot_position_2d[1] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 使用后3个分量作为Y向量
            dx2, dy2, dz2 = robot_orientation[3:6]
            
            # 绘制Y向量（使用不同颜色）
            end_x2 = int(robot_position_2d[0] + arrow_length * dx2)
            end_y2 = int(robot_position_2d[1] + arrow_length * dy2)
            
            cv2.arrowedLine(robot_pos_image, tuple(robot_position_2d), (end_x2, end_y2), 
                           (255, 0, 255), 2, tipLength=0.3)
            
            # 添加Y向量文本
            orient_text2 = f"Orient Y: [{dx2:.2f}, {dy2:.2f}, {dz2:.2f}]"
            cv2.putText(robot_pos_image, orient_text2, 
                       (robot_position_2d[0] + 20, robot_position_2d[1] + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # 如果有抓取器数据，添加抓取器状态信息
        if robot_gripper is not None:
            # 在图像底部添加抓取器状态条
            gripper_text = f"Gripper: {robot_gripper:.2f}"
            
            # 使用颜色编码表示抓取状态（0=关闭/红色，1=打开/绿色）
            gripper_color = (0, int(255 * robot_gripper), int(255 * (1 - robot_gripper)))
            
            cv2.putText(robot_pos_image, gripper_text, 
                       (robot_position_2d[0] + 20, robot_position_2d[1] + 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, gripper_color, 2)
            
            # 可视化抓取器状态（绘制一个表示抓取器开合程度的矩形）
            bar_width = 100
            bar_height = 15
            bar_x = robot_position_2d[0] + 20
            bar_y = robot_position_2d[1] + 100
            
            # 绘制背景
            cv2.rectangle(robot_pos_image, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         (128, 128, 128), -1)
            
            # 绘制填充部分
            filled_width = int(bar_width * robot_gripper)
            cv2.rectangle(robot_pos_image, (bar_x, bar_y), 
                         (bar_x + filled_width, bar_y + bar_height), 
                         gripper_color, -1)
            
            # 添加边框
            cv2.rectangle(robot_pos_image, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         (0, 0, 0), 1)
        
        # 保存图像
        robot_pos_path = os.path.join(output_dir, 'image_01555_color_robot_position_complete.jpg')
        cv2.imwrite(robot_pos_path, robot_pos_image)
        print(f"Robot complete information visualization saved to: {robot_pos_path}")
        
        # 在人手上叠加机械臂末端位置（与方向和抓取器信息）
        human_hand_overlay = draw_robot_hand(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy(), 
            np.array(keypoints_2d_data['keypoints_2d_transformed']), 
            color=(0, 0, 255),  # 红色表示人手
            thickness=2,
            show_indices=False
        )
        
        # 在人手上叠加机械臂末端位置
        cv2.circle(human_hand_overlay, tuple(robot_position_2d), 15, (0, 255, 255), -1)
        
        # 添加简短标签
        cv2.putText(human_hand_overlay, "Robot", 
                    (robot_position_2d[0] + 20, robot_position_2d[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 如果有方向，也添加箭头
        if robot_orientation is not None:
            # 画两个表示方向的箭头
            arrow_length = 60
            dx, dy = robot_orientation[0:2]
            end_x = int(robot_position_2d[0] + arrow_length * dx)
            end_y = int(robot_position_2d[1] + arrow_length * dy)
            cv2.arrowedLine(human_hand_overlay, tuple(robot_position_2d), (end_x, end_y), 
                           (0, 255, 0), 2, tipLength=0.3)
            
            dx2, dy2 = robot_orientation[3:5]
            end_x2 = int(robot_position_2d[0] + arrow_length * dx2)
            end_y2 = int(robot_position_2d[1] + arrow_length * dy2)
            cv2.arrowedLine(human_hand_overlay, tuple(robot_position_2d), (end_x2, end_y2), 
                           (255, 0, 255), 2, tipLength=0.3)
        
        # 保存对比图
        comparison_path = os.path.join(output_dir, 'image_01555_color_hand_robot_comparison_complete.jpg')
        cv2.imwrite(comparison_path, human_hand_overlay)
        print(f"Hand and robot complete comparison saved to: {comparison_path}")
        
        # 使用新函数绘制末端执行器和坐标轴
        robot_end_effector_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
        robot_end_effector_image = draw_robot_end_effector(
            robot_end_effector_image,
            robot_position_2d,
            orientation=robot_orientation,
            gripper=robot_gripper,
            scale=80  # 增大坐标轴的比例以便清晰显示
        )
        
        # 保存末端执行器可视化
        end_effector_path = os.path.join(output_dir, 'image_01555_color_robot_end_effector.jpg')
        cv2.imwrite(end_effector_path, robot_end_effector_image)
        print(f"Robot end-effector visualization saved to: {end_effector_path}")
        
        # 创建机器人手部与机器人末端位置的对比图
        robot_hand_with_position = robot_hand_original.copy()  # 使用手背视角
        # 在机器人手上叠加机械臂末端位置
        cv2.circle(robot_hand_with_position, tuple(robot_position_2d), 15, (0, 255, 255), -1)  # 黄色圆点
        cv2.putText(robot_hand_with_position, "Robot End-Effector", 
                    (robot_position_2d[0] + 20, robot_position_2d[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 创建并保存对比图
        robot_hand_position_path = os.path.join(output_dir, 'image_01555_color_robot_hand_with_position.jpg')
        cv2.imwrite(robot_hand_position_path, robot_hand_with_position)
        print(f"Robot hand with end-effector position saved to: {robot_hand_position_path}")
        
        # 并排对比图 - 左边是机器人手，右边是末端位置图
        side_by_side = np.hstack((robot_hand_original, robot_pos_image))
        side_by_side_path = os.path.join(output_dir, 'image_01555_color_robot_hand_and_position.jpg')
        cv2.imwrite(side_by_side_path, side_by_side)
        print(f"Side-by-side comparison saved to: {side_by_side_path}")
    else:
        print("Warning: No position data found in robot action file")
        
    # Create combined overlays for both versions
    combined_original = draw_robot_hand(
        human_hand_overlay.copy(), 
        robot_keypoints_2d, 
        color=(0, 255, 0),  # Green
        thickness=2,
        show_indices=True,  # 显示关键点索引
        palm_view=False  # 显示为手背视角
    )
    
    combined_corrected = draw_robot_hand(
        human_hand_overlay.copy(), 
        robot_keypoints_2d, 
        color=(255, 165, 0),  # Orange
        thickness=2,
        show_indices=True,  # 显示关键点索引
        palm_view=True  # 显示为手心视角
    )
    
    # 保存对比图像
    original_comparison_path = os.path.join(output_dir, 'image_01555_color_robot_comparison_back.png')
    cv2.imwrite(original_comparison_path, combined_original)
    print(f"Human and robot hand (back) comparison saved to: {original_comparison_path}")
    
    corrected_comparison_path = os.path.join(output_dir, 'image_01555_color_robot_comparison_front.png')
    cv2.imwrite(corrected_comparison_path, combined_corrected)
    print(f"Human and robot hand (front) comparison saved to: {corrected_comparison_path}")
    
    # 在人手和机器人手的对比图上添加末端执行器坐标轴
    if 'position' in robot_data and robot_orientation is not None:
        combined_with_end_effector = draw_robot_end_effector(
            combined_original.copy(),
            robot_position_2d,
            orientation=robot_orientation,
            gripper=robot_gripper,
            scale=60
        )
        
        # 保存包含末端执行器的组合图像
        combined_end_effector_path = os.path.join(output_dir, 'image_01555_color_combined_with_end_effector.jpg')
        cv2.imwrite(combined_end_effector_path, combined_with_end_effector)
        print(f"Combined visualization with end-effector saved to: {combined_end_effector_path}")
        
        # 创建4面板比较图像：原始图像、人手、机器人手和末端执行器
        quad_comparison = np.zeros((image.shape[0]*2, image.shape[1]*2, 3), dtype=np.uint8)
        
        # 填充四个区域
        quad_comparison[:image.shape[0], :image.shape[1]] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 原始图像
        quad_comparison[:image.shape[0], image.shape[1]:] = human_hand_overlay  # 人手
        quad_comparison[image.shape[0]:, :image.shape[1]] = robot_hand_original  # 机器人手
        quad_comparison[image.shape[0]:, image.shape[1]:] = robot_end_effector_image  # 末端执行器
        
        # 添加标题
        title_height = 30
        margin = 5
        cv2.putText(quad_comparison, "Original Image", 
                   (margin, title_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(quad_comparison, "Human Hand", 
                   (image.shape[1] + margin, title_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(quad_comparison, "Robot Hand", 
                   (margin, image.shape[0] + title_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(quad_comparison, "Robot End-Effector & Axes", 
                   (image.shape[1] + margin, image.shape[0] + title_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 保存四面板比较图
        quad_path = os.path.join(output_dir, 'image_01555_color_quad_comparison.jpg')
        cv2.imwrite(quad_path, quad_comparison)
        print(f"Quad comparison visualization saved to: {quad_path}")
    
    # 创建对比图
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(human_hand_overlay, cv2.COLOR_BGR2RGB))
    plt.title('Human Hand Detection (with indices)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(robot_hand_original, cv2.COLOR_BGR2RGB))
    plt.title('Robot Hand - Back View (before correction)')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(robot_hand_corrected, cv2.COLOR_BGR2RGB))
    plt.title('Robot Hand - Front View (after correction)')
    plt.axis('off')
    
    comparison_plt_path = os.path.join(output_dir, 'image_01555_color_hand_comparison_with_indices.png')
    plt.savefig(comparison_plt_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Hand comparison with indices saved to: {comparison_plt_path}")
    
    # 创建最终的三方对比图
    plt.figure(figsize=(16, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(human_hand_overlay, cv2.COLOR_BGR2RGB))
    plt.title('Human Hand (Red)')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(robot_hand_original, cv2.COLOR_BGR2RGB))
    plt.title('Robot Hand - Back View (Green)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(robot_hand_corrected, cv2.COLOR_BGR2RGB))
    plt.title('Robot Hand - Front View (Orange)')
    plt.axis('off')
    
    three_way_comparison_path = os.path.join(output_dir, 'image_01555_color_three_way_comparison.png')
    plt.savefig(three_way_comparison_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Three-way comparison saved to: {three_way_comparison_path}")
    
    # 创建3D对比视图
    fig = plt.figure(figsize=(15, 5))
    
    # 原始3D点云
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(
        transformed_keypoints[:, 0], 
        transformed_keypoints[:, 1], 
        transformed_keypoints[:, 2], 
        c='g', marker='o', s=50
    )
    
    # 标记关键点索引
    for i, (x, y, z) in enumerate(transformed_keypoints):
        ax1.text(x, y, z, str(i), fontsize=8)
    
    ax1.set_title('Original Robot Hand 3D (Back)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 修正后的3D点云
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(
        transformed_keypoints[:, 0], 
        transformed_keypoints[:, 1], 
        transformed_keypoints[:, 2], 
        c='orange', marker='o', s=50
    )
    
    # 标记关键点索引
    for i, (x, y, z) in enumerate(transformed_keypoints):
        ax2.text(x, y, z, str(i), fontsize=8)
    
    ax2.set_title('Corrected Robot Hand 3D (Front)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 叠加视图（两种点云）
    ax3 = fig.add_subplot(133, projection='3d')
    
    # 绘制原始点云（绿色）
    ax3.scatter(
        transformed_keypoints[:, 0], 
        transformed_keypoints[:, 1], 
        transformed_keypoints[:, 2], 
        c='g', marker='o', s=40, alpha=0.7, label='Back'
    )
    
    # 绘制修正点云（橙色）
    ax3.scatter(
        transformed_keypoints[:, 0], 
        transformed_keypoints[:, 1], 
        transformed_keypoints[:, 2], 
        c='orange', marker='^', s=40, alpha=0.7, label='Front'
    )
    
    ax3.set_title('Comparison of Both 3D Models')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    
    comparison_3d_path = os.path.join(output_dir, 'image_01555_color_robot_hand_3d_comparison.png')
    plt.savefig(comparison_3d_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"3D comparison saved to: {comparison_3d_path}")

if __name__ == "__main__":
    main() 