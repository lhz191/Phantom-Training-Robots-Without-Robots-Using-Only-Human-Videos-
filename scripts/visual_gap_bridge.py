import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from typing import Dict, Tuple, List, Optional, Union

class VisualObservationGapBridge:
    """
    Implementation of the visual observation gap bridging component described in section C.
    Adapts the Rovi-Aug data-editing scheme for human-to-robot transfer setting.
    
    This class transforms human demonstration images (Ih,t) to robot images (Ir,t)
    to address the visual gap between human and robot embodiments.
    """
    
    def __init__(
        self,
        robot_gripper_template_path: str,
        human_hand_mask_threshold: float = 0.8,
        blend_alpha: float = 0.9,
        preserve_background: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the visual gap bridge component.
        
        Args:
            robot_gripper_template_path: Path to the robot gripper template image
            human_hand_mask_threshold: Threshold for segmenting human hand
            blend_alpha: Blending factor for overlay of robot gripper
            preserve_background: Whether to preserve background in transformation
            device: Device to perform computations on
        """
        self.device = device
        self.human_hand_mask_threshold = human_hand_mask_threshold
        self.blend_alpha = blend_alpha
        self.preserve_background = preserve_background
        
        # Load robot gripper template if path exists
        if os.path.exists(robot_gripper_template_path):
            self.robot_template = cv2.imread(robot_gripper_template_path)
            if self.robot_template is not None:
                self.robot_template = cv2.cvtColor(self.robot_template, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Could not load robot gripper template from {robot_gripper_template_path}")
        else:
            raise FileNotFoundError(f"Robot gripper template not found at {robot_gripper_template_path}")
        
        # Initialize hand segmentation model (placeholder - needs to be replaced with actual implementation)
        self.hand_segmentation_model = self._init_hand_segmentation_model()
        
        # Initialize transforms for scaling/positioning the robot gripper
        self._init_transforms()
    
    def _init_hand_segmentation_model(self):
        """
        Initialize a model for hand segmentation.
        
        Returns:
            A hand segmentation model (placeholder for now)
        """
        # This is a placeholder - should be replaced with an actual hand segmentation model
        # Could use a pre-trained model like MediaPipe Hands or similar
        return None
    
    def _init_transforms(self):
        """Initialize transforms for scaling and positioning the robot gripper."""
        # These would be learned parameters in the full implementation
        # For now, we'll use default values
        self.scale_factor = 1.0
        self.position_offset_x = 0
        self.position_offset_y = 0
        self.rotation_angle = 0
    
    def segment_hand(self, image: np.ndarray) -> np.ndarray:
        """
        Segment the hand region in the input image.
        
        Args:
            image: Input RGB image containing a human hand
            
        Returns:
            Hand mask (values between 0 and 1)
        """
        # Placeholder implementation - should be replaced with actual hand segmentation
        # In real implementation, this would use self.hand_segmentation_model
        
        # For demo purposes, convert to HSV and use skin color detection
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        # Normalize mask to [0, 1]
        mask = mask / 255.0
        
        return mask
    
    def compute_hand_parameters(self, hand_mask: np.ndarray) -> Dict:
        """
        Compute parameters of hand position, orientation, and grasp width.
        
        Args:
            hand_mask: Binary mask of the hand
            
        Returns:
            Dictionary with hand parameters (position, orientation, grasp_width)
        """
        # Find contours in the mask
        mask_uint8 = (hand_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'position': (0, 0),
                'orientation': 0,
                'grasp_width': 1.0
            }
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute center of mass
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            cx, cy = 0, 0
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        
        # Compute orientation using PCA
        points = largest_contour.reshape(-1, 2).astype(np.float32)
        if len(points) >= 2:  # Need at least 2 points for PCA
            _, _, eigenvectors = cv2.PCACompute2(points, mean=None)
            angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
            orientation = np.degrees(angle)
        else:
            orientation = 0
        
        # Estimate grasp width by finding the width of the contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        grasp_width = min(w, h) / max(mask_uint8.shape)  # Normalize by image size
        
        return {
            'position': (cx, cy),
            'orientation': orientation,
            'grasp_width': grasp_width
        }
    
    def transform_robot_gripper(self, 
                               robot_template: np.ndarray, 
                               hand_params: Dict, 
                               target_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform the robot gripper template based on hand parameters.
        
        Args:
            robot_template: RGB image of the robot gripper template
            hand_params: Parameters extracted from the hand
            target_shape: Target shape (height, width) for the output
            
        Returns:
            Tuple of (transformed_gripper_rgb, gripper_mask)
        """
        # Extract parameters
        target_h, target_w = target_shape
        position = hand_params['position']
        orientation = hand_params['orientation']
        grasp_width = hand_params['grasp_width']
        
        # Scale the template according to grasp width
        scale = grasp_width * self.scale_factor
        h, w = robot_template.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        if new_h <= 0 or new_w <= 0:
            # Fallback to original size if scaling is too small
            new_h, new_w = h, w
            
        resized_template = cv2.resize(robot_template, (new_w, new_h))
        
        # Create rotation matrix
        angle = orientation + self.rotation_angle
        center = (new_w // 2, new_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated_template = cv2.warpAffine(resized_template, M, (new_w, new_h))
        
        # Create a mask for the template (non-zero pixels)
        template_mask = np.any(rotated_template > 0, axis=2).astype(np.float32)
        
        # Create output image of target size
        transformed_gripper = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        gripper_mask = np.zeros((target_h, target_w), dtype=np.float32)
        
        # Calculate target position
        target_x = position[0] + self.position_offset_x
        target_y = position[1] + self.position_offset_y
        
        # Calculate placement bounds
        x_min = max(0, int(target_x - new_w // 2))
        y_min = max(0, int(target_y - new_h // 2))
        x_max = min(target_w, int(target_x + new_w // 2))
        y_max = min(target_h, int(target_y + new_h // 2))
        
        # Calculate source bounds
        src_x_min = max(0, -int(target_x - new_w // 2))
        src_y_min = max(0, -int(target_y - new_h // 2))
        src_x_max = src_x_min + (x_max - x_min)
        src_y_max = src_y_min + (y_max - y_min)
        
        # Place the transformed gripper and mask
        if x_max > x_min and y_max > y_min:
            transformed_gripper[y_min:y_max, x_min:x_max] = rotated_template[
                src_y_min:src_y_max, src_x_min:src_x_max
            ]
            gripper_mask[y_min:y_max, x_min:x_max] = template_mask[
                src_y_min:src_y_max, src_x_min:src_x_max
            ]
        
        return transformed_gripper, gripper_mask
    
    def transform_image(self, human_image: np.ndarray) -> np.ndarray:
        """
        Transform a human demonstration image to a robot image (Ih,t → Ir,t).
        
        Args:
            human_image: RGB image containing a human hand/arm
            
        Returns:
            Transformed image with robot gripper replacing the human hand
        """
        # Segment the hand
        hand_mask = self.segment_hand(human_image)
        
        # Compute hand parameters
        hand_params = self.compute_hand_parameters(hand_mask)
        
        # Transform the robot gripper template
        transformed_gripper, gripper_mask = self.transform_robot_gripper(
            self.robot_template, hand_params, human_image.shape[:2]
        )
        
        # Create the final image by blending
        if self.preserve_background:
            # Invert the hand mask to get the background
            background_mask = 1.0 - (hand_mask * self.human_hand_mask_threshold)
            background_mask = np.clip(background_mask, 0, 1)
            
            # Extract background from original image
            background = human_image * background_mask[:, :, np.newaxis]
            
            # Blend robot gripper with background
            robot_image = background.copy()
            mask_3d = gripper_mask[:, :, np.newaxis]
            robot_image = (1 - mask_3d) * robot_image + mask_3d * transformed_gripper
            
            # Apply final blending
            robot_image = robot_image.astype(np.uint8)
        else:
            # Simply overlay the transformed gripper
            mask_3d = gripper_mask[:, :, np.newaxis]
            robot_image = (1 - mask_3d) * human_image + mask_3d * transformed_gripper
            robot_image = robot_image.astype(np.uint8)
        
        return robot_image
    
    def transform_batch(self, human_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Transform a batch of human demonstration images to robot images.
        
        Args:
            human_images: List of RGB images containing human hands/arms
            
        Returns:
            List of transformed images with robot gripper
        """
        return [self.transform_image(img) for img in human_images]
    
    def __call__(self, human_images: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Call method for easy use of the class.
        
        Args:
            human_images: Single image or list of images
            
        Returns:
            Transformed image(s)
        """
        if isinstance(human_images, list):
            return self.transform_batch(human_images)
        else:
            return self.transform_image(human_images)


class RoviAugAdapter(nn.Module):
    """
    Neural network module that implements the Rovi-Aug adaptation for human-to-robot transfer.
    
    This module combines hand segmentation, robot gripper rendering, and image blending
    in a differentiable way that can be trained end-to-end.
    """
    
    def __init__(
        self,
        robot_gripper_template_path: str,
        use_pretrained_segmentation: bool = True,
        trainable: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the RoviAug adapter module.
        
        Args:
            robot_gripper_template_path: Path to the robot gripper template image
            use_pretrained_segmentation: Whether to use a pretrained segmentation model
            trainable: Whether the module is trainable
            device: Device to perform computations on
        """
        super(RoviAugAdapter, self).__init__()
        self.device = device
        self.trainable = trainable
        
        # Load robot gripper template
        if os.path.exists(robot_gripper_template_path):
            robot_template = cv2.imread(robot_gripper_template_path)
            if robot_template is not None:
                robot_template = cv2.cvtColor(robot_template, cv2.COLOR_BGR2RGB)
                # Convert to tensor [C, H, W]
                robot_template = torch.from_numpy(robot_template.transpose(2, 0, 1)).float() / 255.0
                self.register_buffer('robot_template', robot_template)
            else:
                raise ValueError(f"Could not load robot gripper template from {robot_gripper_template_path}")
        else:
            raise FileNotFoundError(f"Robot gripper template not found at {robot_gripper_template_path}")
        
        # Hand segmentation model
        self.segmentation_model = self._init_segmentation_model(use_pretrained_segmentation)
        
        # Transformation parameters (learnable if trainable=True)
        self.scale_factor = nn.Parameter(torch.tensor(1.0), requires_grad=trainable)
        self.position_offset_x = nn.Parameter(torch.tensor(0.0), requires_grad=trainable)
        self.position_offset_y = nn.Parameter(torch.tensor(0.0), requires_grad=trainable)
        self.rotation_angle = nn.Parameter(torch.tensor(0.0), requires_grad=trainable)
        self.blend_alpha = nn.Parameter(torch.tensor(0.9), requires_grad=trainable)
    
    def _init_segmentation_model(self, use_pretrained: bool) -> nn.Module:
        """
        Initialize a hand segmentation model.
        
        Args:
            use_pretrained: Whether to use a pretrained model
            
        Returns:
            Hand segmentation model
        """
        # This is a placeholder - should be replaced with an actual segmentation model
        # such as DeepLabV3, UNet, or another segmentation architecture
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Upsampling path
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Set model to non-trainable if required
        if not self.trainable or use_pretrained:
            for param in model.parameters():
                param.requires_grad = False
        
        return model
    
    def compute_hand_parameters(self, hand_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute hand parameters (position, orientation, grasp width) from the segmentation mask.
        
        Args:
            hand_mask: Segmentation mask tensor [B, 1, H, W]
            
        Returns:
            Dictionary with hand parameters as tensors
        """
        batch_size = hand_mask.shape[0]
        device = hand_mask.device
        
        # Convert mask to binary using threshold
        binary_mask = (hand_mask > 0.5).float()
        
        # Initialize results
        positions = torch.zeros(batch_size, 2, device=device)
        orientations = torch.zeros(batch_size, device=device)
        grasp_widths = torch.ones(batch_size, device=device)
        
        # Process each image in the batch
        for b in range(batch_size):
            mask = binary_mask[b, 0].cpu().numpy()
            
            # Skip if mask is empty
            if not np.any(mask):
                continue
            
            # Find connected components
            _, labeled_mask, stats, centroids = cv2.connectedComponentsWithStats(
                (mask * 255).astype(np.uint8), connectivity=8
            )
            
            # Skip background (index 0)
            if len(stats) <= 1:
                continue
            
            # Find largest component (excluding background)
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            
            # Get position (centroid)
            positions[b, 0] = torch.tensor(centroids[largest_idx][0], device=device)
            positions[b, 1] = torch.tensor(centroids[largest_idx][1], device=device)
            
            # Get orientation and grasp width using PCA
            y_indices, x_indices = np.where(labeled_mask == largest_idx)
            if len(y_indices) > 10:  # Need enough points for PCA
                points = np.column_stack([x_indices, y_indices]).astype(np.float32)
                
                # Compute PCA
                mean, eigenvectors = cv2.PCACompute2(points, mean=np.mean(points, axis=0))[0:2]
                
                # Compute orientation from principal eigenvector
                angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
                orientations[b] = torch.tensor(angle, device=device)
                
                # Estimate grasp width from eigenvalues
                eigenvalues, _ = np.linalg.eig(np.cov(points.T))
                grasp_width = np.sqrt(eigenvalues.min()) / mask.shape[0]  # Normalize by image height
                grasp_widths[b] = torch.tensor(grasp_width, device=device)
        
        return {
            'positions': positions,
            'orientations': orientations,
            'grasp_widths': grasp_widths
        }
    
    def spatial_transformer(self, 
                          template: torch.Tensor, 
                          positions: torch.Tensor,
                          orientations: torch.Tensor,
                          grasp_widths: torch.Tensor,
                          target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Apply spatial transformer to position and orient the robot gripper.
        
        Args:
            template: Robot gripper template tensor [C, H, W]
            positions: Position tensors [B, 2]
            orientations: Orientation tensors [B]
            grasp_widths: Grasp width tensors [B]
            target_size: Target size (H, W) for the output images
            
        Returns:
            Transformed robot gripper tensors [B, C, H, W]
        """
        batch_size = positions.shape[0]
        device = positions.device
        h_target, w_target = target_size
        
        # Create batch of template images
        c, h_template, w_template = template.shape
        templates = template.expand(batch_size, c, h_template, w_template)
        
        # Compute scaling factors based on grasp width
        scale = grasp_widths * self.scale_factor
        
        # Create transformation matrices
        theta = torch.zeros(batch_size, 2, 3, device=device)
        
        for b in range(batch_size):
            # Rotation matrix
            angle = orientations[b] + self.rotation_angle
            cos_val = torch.cos(angle)
            sin_val = torch.sin(angle)
            
            # Scale factors
            s = scale[b]
            
            # Set rotation and scaling
            theta[b, 0, 0] = cos_val * s
            theta[b, 0, 1] = sin_val * s
            theta[b, 1, 0] = -sin_val * s
            theta[b, 1, 1] = cos_val * s
            
            # Set translation to center at the target position
            # Convert position from pixel coordinates to normalized [-1, 1] coordinates
            pos_x = 2.0 * (positions[b, 0] + self.position_offset_x) / w_target - 1.0
            pos_y = 2.0 * (positions[b, 1] + self.position_offset_y) / h_target - 1.0
            
            theta[b, 0, 2] = pos_x
            theta[b, 1, 2] = pos_y
        
        # Create sampling grid
        grid = F.affine_grid(theta, size=(batch_size, c, h_target, w_target), align_corners=False)
        
        # Apply grid sampling
        transformed = F.grid_sample(templates, grid, align_corners=False, mode='bilinear')
        
        return transformed
    
    def forward(self, human_images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: transform human images to robot images.
        
        Args:
            human_images: Input tensor of human demonstration images [B, C, H, W]
            
        Returns:
            Transformed robot images [B, C, H, W]
        """
        batch_size, channels, height, width = human_images.shape
        
        # Segment the hand
        hand_mask = self.segmentation_model(human_images)
        
        # Compute hand parameters
        hand_params = self.compute_hand_parameters(hand_mask)
        
        # Transform robot gripper template
        transformed_gripper = self.spatial_transformer(
            self.robot_template,
            hand_params['positions'],
            hand_params['orientations'],
            hand_params['grasp_widths'],
            (height, width)
        )
        
        # Create gripper mask (any non-zero channel)
        gripper_mask = (transformed_gripper.sum(dim=1, keepdim=True) > 0.01).float()
        
        # Create the final image by blending
        # Invert the hand mask to get the background
        background_mask = 1.0 - (hand_mask * 0.8)
        background_mask = torch.clamp(background_mask, 0, 1)
        
        # Extract background from original image
        background = human_images * background_mask
        
        # Blend robot gripper with background
        robot_images = background * (1 - gripper_mask) + transformed_gripper * gripper_mask
        
        return robot_images


def test_visual_gap_bridge():
    """Test function for the VisualObservationGapBridge class."""
    import matplotlib.pyplot as plt
    
    # Create a test image folder
    test_folder = "test_images"
    os.makedirs(test_folder, exist_ok=True)
    
    # Create a dummy robot gripper template (red rectangle)
    gripper_template = np.zeros((100, 50, 3), dtype=np.uint8)
    gripper_template[:, :, 0] = 255  # Red color
    cv2.rectangle(gripper_template, (10, 10), (40, 90), (0, 0, 255), 2)
    cv2.rectangle(gripper_template, (10, 10), (40, 50), (0, 255, 0), 2)
    template_path = os.path.join(test_folder, "robot_gripper_template.png")
    cv2.imwrite(template_path, cv2.cvtColor(gripper_template, cv2.COLOR_RGB2BGR))
    
    # Create a dummy human hand image (skin colored ellipse)
    human_image = np.zeros((200, 300, 3), dtype=np.uint8)
    human_image[:] = (120, 120, 120)  # Gray background
    cv2.ellipse(human_image, (150, 100), (60, 40), 45, 0, 360, (200, 150, 140), -1)  # Skin colored ellipse
    cv2.line(human_image, (120, 70), (180, 130), (180, 130, 100), 10)  # Finger
    human_image_path = os.path.join(test_folder, "human_hand.png")
    cv2.imwrite(human_image_path, cv2.cvtColor(human_image, cv2.COLOR_RGB2BGR))
    
    # Initialize the bridge
    bridge = VisualObservationGapBridge(template_path)
    
    # Transform the image
    robot_image = bridge.transform_image(human_image)
    
    # Save the result
    result_path = os.path.join(test_folder, "robot_transformed.png")
    cv2.imwrite(result_path, cv2.cvtColor(robot_image, cv2.COLOR_RGB2BGR))
    
    # Display the results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(human_image)
    plt.title("Human Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(gripper_template)
    plt.title("Robot Gripper Template")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(robot_image)
    plt.title("Transformed Robot Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(test_folder, "comparison.png"))
    plt.close()
    
    print(f"Test images saved to {test_folder}")
    
    return bridge


if __name__ == "__main__":
    test_visual_gap_bridge() 