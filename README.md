# SpeedLens: Vehicle Speed Estimation, Tracking and Counting

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## 1. Introduction

**SpeedLens** is an advanced vehicle tracking, counting, and speed estimation system that combines state-of-the-art computer vision models to provide comprehensive traffic analysis. Created by **Siddharth Chakraborty**, this project enables real-time monitoring of traffic flow, vehicle speed measurement, congestion analysis, and enhanced road safety insights.

### Key Features

- **Real-time Vehicle Detection**: Utilizes YOLOv8x for accurate detection of cars, motorcycles, buses, and trucks
- **Multi-Object Tracking**: Employs ByteTrack algorithm for robust tracking across frames
- **Speed Estimation**: Calculates vehicle speeds using perspective transformation and real-world coordinate mapping
- **Vehicle Counting**: Implements line-based counting zones to track vehicles entering/exiting regions
- **TensorRT Acceleration**: Optional TensorRT export for optimized inference performance
- **FPS Monitoring**: Real-time frames-per-second display for performance tracking

### Main Components

- **YOLOv8**: The YOLOv8x model from Ultralytics provides accurate and real-time vehicle detection
- **ByteTrack**: Multi-object tracking algorithm ensuring smooth and reliable tracking of vehicles across frames
- **Perspective Transformation**: Converts pixel coordinates to real-world measurements for accurate speed calculation
- **Line Counter**: Supervision library integration for counting vehicles crossing designated zones
- **TensorRT Inference**: Optional model export to TensorRT format for accelerated inference

## 2. Installation

To set up SpeedLens, follow these steps to configure your environment with the required dependencies:

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for TensorRT acceleration)

### Setup Steps

1. **Clone the repository:**
```bash
   git clone https://github.com/yourusername/speedlens.git
```

2. **Navigate to the repository and install dependencies:**
```bash
   cd speedlens
   pip install -r requirements.txt
```

3. **Clone the ByteTrack library inside the current repo:**
```bash
   git clone https://github.com/ifzhang/ByteTrack.git
```
    
4. **Install ByteTrack dependencies:**
```bash
   cd ByteTrack
   
   # Workaround for compatibility issue
   sed -i 's/onnx==1.8.1/onnx==1.9.0/g' requirements.txt

   pip install -r requirements.txt
   python3 setup.py develop
   pip install cython_bbox onemetric loguru lap thop
   
   cd ..
```

5. **Configure speed estimation parameters:**
   
   Edit `speed_config.py` to set up the perspective transformation points that match your camera view. Define source points (pixel coordinates) and target points (real-world coordinates in meters).

## 3. Usage

### Running Vehicle Tracking and Speed Estimation

1. **Configure the pipeline** by editing `main.py`:
```python
   input_video = "assets/vehicle-counting.mp4"      # Path to input video
   output_video = "assets/vehicle-counting-result.mp4"  # Path to save output
   use_tensorrt = False  # Set to True for TensorRT acceleration
```

2. **Adjust the counting line** (optional):
   
   In the `vehicle_tracker_and_counter` class initialization, modify:
```python
   self.line_start = sv.Point(50, 1500)
   self.line_end = sv.Point(3840-50, 1500)
```

3. **Run the detection, tracking, and speed estimation pipeline:**
```bash
   python main.py
```

4. **Output**: The processed video will be saved to the specified `output_video` path, displaying:
   - Bounding boxes around detected vehicles
   - Tracker IDs for each vehicle
   - Real-time speed estimates (km/h)
   - Vehicle counts crossing the line
   - FPS performance metrics

### Example Output

<p align="center">
  <img src="assets/output_video.PNG" width="700" title="SpeedLens Output Frame">
</p>

The output frame shows tracked vehicles with unique IDs, their estimated speeds in km/h, confidence scores, and the counting line with vehicle totals.

## 4. Project Structure
```
speedlens/
├── main.py                 # Main pipeline script
├── utils.py                # Utility functions for tracking
├── speed_config.py         # Perspective transformation configuration
├── requirements.txt        # Python dependencies
├── assets/                 # Input/output videos and images
├── ByteTrack/             # ByteTrack submodule
└── README.md              # This file
```

## 5. How It Works

### Detection
YOLOv8x detects vehicles in each frame, filtering for specific classes (cars, motorcycles, buses, trucks).

### Tracking
ByteTrack associates detections across frames, assigning unique IDs to maintain vehicle identity.

### Speed Estimation
1. Bottom-center points of bounding boxes are extracted
2. Points are transformed to real-world coordinates using perspective transformation
3. Vehicle displacement is tracked over time
4. Speed is calculated as distance/time and converted to km/h

### Counting
A virtual line zone counts vehicles as they cross the designated threshold.

## 6. Customization

### Adjust Detection Classes
Modify `self.CLASS_ID` in `main.py` to detect different vehicle types:
```python
self.CLASS_ID = [2, 3, 5, 7]  # car, motorcycle, bus, truck
```

### Configure Speed Calculation
Edit `speed_config.py` to calibrate the perspective transformation for your specific camera setup.

### TensorRT Optimization
Enable TensorRT for faster inference:
```python
use_tensorrt = True
```

## 7. Performance

- **Detection**: YOLOv8x provides high accuracy with real-time performance
- **Tracking**: ByteTrack maintains stable tracking even in crowded scenes
- **Speed Estimation**: Accurate speed measurement within calibrated zones
- **FPS**: Displayed in real-time on output video

## 8. Author

**Siddharth Chakraborty**

## 9. References

This project builds upon and integrates several outstanding open-source projects:

- **Roboflow Tutorial**: Based on the [vehicle tracking and counting notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-and-count-vehicles-with-yolov8.ipynb)
- **YOLOv8**: [Ultralytics YOLOv8 Repository](https://github.com/ultralytics/ultralytics)
- **ByteTrack**: [ByteTrack Repository](https://github.com/ifzhang/ByteTrack) by Yifu Zhang et al.
- **Supervision**: [Supervision Library](https://github.com/roboflow/supervision) by Roboflow

## 10. License

This project is provided as-is for educational and research purposes. Please refer to the individual licenses of YOLOv8, ByteTrack, and Supervision libraries for commercial use.

## 11. Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

---

**SpeedLens** - Comprehensive traffic analysis powered by computer vision