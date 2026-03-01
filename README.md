VisionCity: Real-Time Urban Intelligence System
VisionCity is a high-performance computer vision pipeline designed for smart city infrastructure. By leveraging state-of-the-art deep learning models, it transforms standard surveillance feeds into actionable urban analytics, enabling cities to monitor traffic flow, pedestrian safety, and infrastructure utilization in real-time.

Key Features:

High-Speed Detection: Powered by YOLOv8, achieving 30+ FPS on edge-compatible hardware.

Robust Tracking: Implements the Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) algorithm to maintain object IDs across occlusions and frame gaps.

Granular Classification: Distinguishes between cars, heavy-duty trucks, motorcycles, and bicycles.

Crowd Dynamics: Real-time pedestrian counting and density mapping to identify bottlenecks in walkways.

Spatial Analytics: Generates dynamic heatmaps to visualize long-term traffic patterns and "near-miss" hotspots.

Edge-Ready API: A lightweight FastAPI wrapper allows for seamless integration with IoT dashboards and centralized city management software.

System Architecture:

The system is designed for low-latency inference by processing video streams at the edge and pushing only the metadata (coordinates, IDs, and classes) to a central database.

Ingestion: RTSP/HTTP video stream capture.

Tech Stack
Component
Technology
Model
Ultralytics YOLOv8 (Custom Trained)
Tracking
Deep SORT
Backend
Python 3.9+, FastAPI
Inference 
EngineONNX Runtime / TensorRT (Optimized for Edge)
Database
Redis (for real-time metrics) / PostgreSQL

Inference: YOLOv8 extracts bounding boxes and class probabilities.

Tracking: Deep SORT associates detections across frames using Kalman filtering and Re-ID features.

Analytics: Logic layer calculates flow rates and updates the density heatmap.

Egress: Data is served via REST API or WebSockets for real-time visualization.

Getting Started

Prerequisites
--Python 3.9+

--NVIDIA GPU with CUDA 11.x (Recommended for 30+ FPS)

--FFmpeg

Installation:
1. Clone the repository:
   git clone https://github.com/your-username/visioncity.git
cd visioncity

2.Install dependencies:
    pip install -r requirements.txt

3.Download Model Weights:
    Place your custom best.pt or yolov8n.pt in the /models directory.

Running the System:
To start the FastAPI server and process a local video file or RTSP stream:
--python main.py --source "path/to/video.mp4" --show-live True

Analytics & Visualization:
Heatmap Generation
The system accumulates trajectory data to highlight high-traffic zones. This is vital for urban planners to decide where to implement bike lanes or adjust signal timings.

API Endpoints:
GET /analytics/live: Returns current object counts (pedestrians, vehicles).

GET /analytics/history: Returns time-series data for traffic flow.

POST /stream/start: Connect to a new RTSP camera feed
Performance Benchmarking
Device Precision (mAP 50-95) Inference Speed
NVIDIA Jetson Orin 0.48  ~35
FPSNVIDIA RTX 30600  52~90 
FPSCPU (i7-10th Gen) 0.45~ 5 FPS

Contributing:
Contributions are what make the open-source community such an amazing place to learn, inspire, and create.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request
