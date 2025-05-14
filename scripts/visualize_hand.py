import numpy as np
import json
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def visualize_hand_data(obj_file, keypoints_file):
    """
    Visualize hand model and keypoints
    
    Parameters:
    obj_file: OBJ file path containing hand mesh model
    keypoints_file: JSON file path containing keypoint data
    """
    # Load 3D model
    print(f"Loading model file: {obj_file}")
    mesh = trimesh.load(obj_file)
    vertices = np.array(mesh.vertices)
    
    # Load keypoint data
    print(f"Loading keypoints file: {keypoints_file}")
    with open(keypoints_file, 'r') as f:
        keypoints_data = json.load(f)
    
    # Prioritize using transformed keypoints (if they exist)
    if 'keypoints_3d_transformed' in keypoints_data:
        print("Using transformed keypoints (aligned with OBJ model coordinate system)")
        keypoints = np.array(keypoints_data['keypoints_3d_transformed'])
    else:
        print("Warning: No transformed keypoints found, using original coordinates (may not align with OBJ model)")
        keypoints = np.array(keypoints_data['keypoints_3d'])
    
    # 自动将keypoints_3d（如果存在）用camera_translation和平移+180度绕X轴旋转变换，并保存
    if 'keypoints_3d' in keypoints_data and 'camera_translation' in keypoints_data:
        keypoints_3d_raw = np.array(keypoints_data['keypoints_3d'])
        camera_translation = np.array(keypoints_data['camera_translation'])
        # 平移
        keypoints_3d_trans = keypoints_3d_raw + camera_translation
        # 180度绕X轴旋转
        rot_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        keypoints_3d_trans = np.dot(keypoints_3d_trans, rot_matrix.T)
        # 保存到txt
        out_txt = os.path.splitext(keypoints_file)[0] + '_3d_transformed_from_raw.txt'
        np.savetxt(out_txt, keypoints_3d_trans, fmt='%.6f')
        print(f"Transformed keypoints_3d (from raw) saved to: {out_txt}")
    
    is_right = keypoints_data['is_right']
    
    # Print information
    print(f"Model vertices count: {len(vertices)}")
    print(f"Keypoints count: {len(keypoints)}")
    print(f"Is right hand? {'Yes' if is_right else 'No'}")
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw model vertices (only taking a subset to improve performance)
    subsample = 1  # Take every subsample-th point
    ax.scatter(vertices[::subsample, 0], 
               vertices[::subsample, 1], 
               vertices[::subsample, 2], 
               c='blue', alpha=0.1, s=1, label='Model Vertices')
    
    # Draw keypoints
    ax.scatter(keypoints[:, 0], 
               keypoints[:, 1], 
               keypoints[:, 2], 
               c='red', s=50, label='Keypoints')
    
    # Add labels to keypoints
    for i, (x, y, z) in enumerate(keypoints):
        ax.text(x, y, z, str(i), fontsize=8)
    
    # Connect keypoints to show hand structure
    # Define connections between finger keypoints
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Little finger
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    for start, end in connections:
        ax.plot([keypoints[start, 0], keypoints[end, 0]],
                [keypoints[start, 1], keypoints[end, 1]],
                [keypoints[start, 2], keypoints[end, 2]], 'green')
    
    # Set figure properties
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(f"Hand Model ({len(vertices)} vertices) and Keypoints ({len(keypoints)} points) Visualization")
    plt.legend()
    
    # Save image
    output_file = os.path.splitext(obj_file)[0] + '_visualization.png'
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to: {output_file}")
    
    # Show figure
    plt.show()
    
    # Create second figure: only showing keypoints
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw keypoints
    ax.scatter(keypoints[:, 0], 
               keypoints[:, 1], 
               keypoints[:, 2], 
               c='red', s=80, label='Keypoints')
    
    # Add labels to keypoints
    for i, (x, y, z) in enumerate(keypoints):
        ax.text(x, y, z, str(i), fontsize=10)
    
    # Connect keypoints to show hand structure
    for start, end in connections:
        ax.plot([keypoints[start, 0], keypoints[end, 0]],
                [keypoints[start, 1], keypoints[end, 1]],
                [keypoints[start, 2], keypoints[end, 2]], 'green')
    
    # Set figure properties
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(f"Hand Keypoints ({len(keypoints)} points) Visualization")
    
    # Save keypoints image
    keypoints_output_file = os.path.splitext(obj_file)[0] + '_keypoints.png'
    plt.savefig(keypoints_output_file, dpi=300)
    print(f"Keypoints visualization saved to: {keypoints_output_file}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize hand model and keypoints')
    parser.add_argument('--obj', type=str, required=True, help='OBJ file path')
    parser.add_argument('--keypoints', type=str, required=True, help='Keypoints JSON file path')
    
    args = parser.parse_args()
    
    visualize_hand_data(args.obj, args.keypoints) 