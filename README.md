# RoadTrack: Highway Abnormal Parking Event Detection Dataset and Benchmarks

## 📖 Overview

This project provides the first publicly available highway abnormal parking event dataset and baseline implementation. The dataset contains 1,000 video sequences, including 200 abnormal parking event videos and 800 normal traffic videos, providing a reliable training foundation and performance evaluation benchmark for highway abnormal event detection research.

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



## 📁 Dataset Structure

```
datasets/
├── P/                          # Main dataset directory
│   ├── 04-12_09_K221_1953/     # Video sequence (date_time_road_frame format)
│   │   └── data/
│   │       ├── det/            # Object detection results
│   │       └── gt/             # Ground truth annotations
│   ├── 04-12_11_K249_680/
│   ├── ...                     # More video sequences
│   └── 06-10_20_K287_172/
```

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

```bibtex

```




## 📧 Contact

For questions or suggestions, please contact us through:

- Email: [corfyi@csust.edu.cn]

## 🙏 Acknowledgments

Thanks to all researchers and developers who contributed to dataset annotation and algorithm development.

---

**Keywords**: Highway Monitoring, Abnormal Parking Detection, Multi-Object Tracking, RoadTrack, Computer Vision, Intelligent Transportation Systems
