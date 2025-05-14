# Robot Hand Project

This project focuses on robot hand control and visualization, including hand pose estimation, transformation, and visualization tools.

## Project Structure

```
.
├── sam2/                      # SAM model integration
├── hamer/                     # Core HAMER implementation
├── E2FGVI/                    # Video inpainting module
├── eDO_description/          # Robot description files
├── scripts/                  # Utility scripts
│   ├── visualize_hand.py
│   ├── robot_renderer.py
│   └── ...
└── docs/                     # Documentation and diagrams
    └── *.drawio             # Architecture diagrams
```

## Features

- Hand pose estimation and tracking
- 3D to 2D projection of hand keypoints
- Robot hand visualization
- Hand transformation between different coordinate systems
- Real-time robot control interface

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/robot_hand_project.git
cd robot_hand_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage examples:

```python
# Example code for hand visualization
from visualize_hand import draw_robot_hand

# Load and visualize hand keypoints
keypoints_2d = ...  # Your 2D keypoints
image = draw_robot_hand(image, keypoints_2d)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- HAMER project for the base implementation
- SAM for segmentation capabilities
- E2FGVI for video inpainting
