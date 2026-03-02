# RoadTrack: Real-Time Distant Parking Detection via Visual Geometric Projection and Speed Estimation

## 📖 Overview

This work has been published in *Signal, Image and Video Processing*.

This project provides the first publicly available distant highway abnormal parking event dataset and baseline implementation. The dataset contains 1,000 video sequences, including 200 abnormal parking event videos and 800 normal traffic videos, providing a reliable training foundation and performance evaluation benchmark for highway abnormal event detection research.

## 🚀 Key Contributions

###  Highway Abnormal Parking Event Dataset
- **Dataset Scale**: 1,000 video sequences
- **Data Distribution**: 200 abnormal event videos + 800 normal traffic videos
- **Annotation Quality**: Provides detailed object detection and tracking annotations
- **Research Value**: Provides standardized evaluation benchmarks for subsequent research

## 🎬 Visualization

The following GIF animations demonstrate the detection process of vehicles transitioning from motion to stationary states at long distances. Green bounding boxes indicate moving vehicles, while red bounding boxes highlight stationary (parked) vehicles that are detected as abnormal parking events.

<div align="center">

| Event 1 | Event 2 |
|:-------:|:-------:|
| ![Stop Car Event 1](gif/stop_car_event1.gif) | ![Stop Car Event 2](gif/stop_car_event2.gif) |

| Event 3 | Event 4 |
|:-------:|:-------:|
| ![Stop Car Event 3](gif/stop_car_event3.gif) | ![Stop Car Event 4](gif/stop_car_event4.gif) |


</div>

- 🚗 **Green Boxes**: Moving vehicles being tracked
- 🚨 **Red Boxes**: Stationary vehicles detected as abnormal parking events
- 📏 **Long-distance Detection**: Effective detection capability beyond 100 meters
- 🎯 **Real-time Tracking**: Continuous monitoring of vehicle status changes

## 📊 Dataset Comparison

The following table presents a comparison of RoadTrack dataset with other existing datasets:

| Dataset | Number of Videos | Number of Events | Long Distance |
|---------|:----------------:|:----------------:|:-------------:|
| HighD | 60 | N/A | × |
| ISLab-PVD | 16 | 22 | × |
| DAD | 668 | 620 | × |
| **RoadTrack** | **1000** | **300** | **√** |

**Our Advantages**
- 📈 **Largest Scale**: 1000 video sequences, providing comprehensive coverage
- 🎯 **Event Focus**: 300 abnormal parking events with detailed annotations
- 🔭 **Long-distance Capability**: Unique focus on long-distance detection (>100m)


## 📁 Dataset Structure



```
datasets/
├── P/                          # Abnormal parking event videos
│   ├── 04-12_09_K221_1953/     # Video sequence (date_time_road_frame format)
│   │   └── data/
│   │       ├── det/            # Object detection results
│   │       └── gt/             # Ground truth annotations
│   ├── ...                     # More abnormal event sequences
├── N/                          # Normal traffic videos
│   ├── 04-12_09_K221_1953/     # Video sequence (date_time_road_frame format)
│   │   └── data/
│   │       ├── det/            # Object detection results
│   │       └── gt/             # Ground truth annotations
│   ├── ...                     # More normal traffic sequences
```

**Video Data Access:**
To download the full RoadTrack video dataset, please apply for access via the following link:
[Download RoadTrack Video Data](https://docs.google.com/forms/d/e/1FAIpQLSfexdOzSz_kMgp7_ctYRjfPOlfvNVoATGkg9ihLK-YoEvqlZQ/viewform)

### Data Format Description

- **Detection File Format** (`det/`): `frame_id,track_id,x1,y1,w,h,conf,class_id`
- **Ground Truth Format** (`gt/`): `frame_id,track_id,x1,y1,w,h,conf,class_id`
- **Coordinate System**: Image coordinate system with top-left as origin, `(x1,y1)` is top-left corner of bounding box, `w,h` are width and height

## 🛠️ Environment Setup

### System Requirements
- Python 3.7+
- OpenCV
- NumPy
- SciPy

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependency List
```
numpy
filterpy
lap
scipy
argparse
opencv-python
```

## 🚀 Quick Start


### 1. Run Baseline Method

```bash
python run_baseline.py
```


### 2. Run RoadTrack Method

```bash
python run_roadtrack.py
```

### 3. Run YOLO Video Detection Demo


We provide a demo script in the `yolo_detector` folder for object detection in videos using the YOLO model. The provided `yolo11m.pt` is the weight file used in our paper, and `video.mp4` is a sample test video from our dataset. For access to the complete video dataset, authorization is required.

**Usage:**

1. The `yolo_detector` directory already contains:
   - `yolo11m.pt`: YOLO model weights used in our publication
   - `video.mp4`: A sample test video from the RoadTrack dataset
2. Run the demo script:

	```bash
	cd yolo_detector
	python run_yolo_demo.py
	```

The script will process the sample video and display detection results in real time using OpenCV. You can press `q` to exit visualization.




## 📊 Performance Evaluation

### Evaluation Metrics

- **Detection Precision**: Abnormal event detection precision
- **Detection Recall**: Abnormal event detection recall
- **F1**: Abnormal event detection F1 Score
- **Detection Latency**: Abnormal event detection latency

### Baseline Performance
Baseline method performance on test set:
- Long-distance parking detection capability: >100m
- Real-time processing capability: Supports real-time video stream analysis



## 🎯 Applications

- **Intelligent Transportation Systems**: Real-time monitoring of highway abnormal events
- **Traffic Safety Management**: Automatic detection and warning of parking events
- **Academic Research**: Multi-object tracking and abnormal detection algorithm research
- **Engineering Applications**: Traffic monitoring system development and deployment

## 📖 Citation

If you use this dataset or method in your research, please cite:

Yi K, Luo L, Xie C, et al. RoadTrack: Real-Time Distant Parking Detection via Visual Geometric Projection and Speed Estimation[J]. Signal, Image and Video Processing, 2026, 20(2): 79.

```bibtex
@article{yi2026roadtrack,
  title={RoadTrack: Real-Time Distant Parking Detection via Visual Geometric Projection and Speed Estimation},
  author={Yi, Kefu and Luo, Longbing and Xie, Chunlin and Zhu, Yufan and Tian, Yunyang},
  journal={Signal, Image and Video Processing},
  volume={20},
  number={2},
  pages={79},
  year={2026},
  publisher={Springer}
}
```




## 📧 Contact

For questions or suggestions, please contact us through:

- Email: [corfyi@csust.edu.cn]

## 🙏 Acknowledgments

Thanks to all researchers and developers who contributed to dataset annotation and algorithm development.

---

**Keywords**: Highway Monitoring, Abnormal Parking Detection, Multi-Object Tracking, RoadTrack, Computer Vision, Intelligent Transportation Systems
