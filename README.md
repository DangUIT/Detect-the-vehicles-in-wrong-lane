Dưới đây là phiên bản sửa lại với thông tin bạn cung cấp:  

---

# Detect Vehicle in Wrong Lane

This project utilizes **computer vision** and **deep learning** techniques to detect vehicles traveling in the wrong lane. The model is based on **YOLOv8n**, optimized for efficient inference using quantization techniques. The project also includes object counting to accurately detect and report lane violations.  

## Table of Contents
- [Overview](#overview)  
- [Features](#features)  
- [Technologies Used](#technologies-used)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Optimization](#model-optimization)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)  

## Overview
The project aims to detect vehicles driving in the wrong lane using a custom-trained **YOLOv8n** model. It includes:  
1. **Model Training:** Training a custom vehicle dataset using **YOLOv8n**.  
2. **Model Optimization:** 
   - **Post-training Quantization (PTQ)** to reduce memory usage and inference time.  
   - **Quantization Aware Training (QAT)** to minimize accuracy loss while reducing computation costs.  
3. **Detection and Counting:** Implementing YOLO-based object counting for accurate detection and tracking of lane violations.  

## Features
- **Real-time Detection:** Detects vehicles in the wrong lane with high accuracy.  
- **Optimized Model:** Faster inference using PTQ and QAT.  
- **Object Counting:** Tracks and counts vehicles for detailed reports.  
- **Scalable:** Adaptable to various traffic scenarios and datasets.  

## Technologies Used
- **YOLOv8n** – for model training and object detection.  
- **Ultralytics** – Python package for YOLOv8.  
- **Python** – main programming language.  
- **OpenCV** – for image and video processing.  
- **CUDA** – optional, for GPU acceleration.  

## Installation
### 1. Clone the Repository
   ```bash
   git clone https://github.com/DangUIT/Detect-the-vehicles-in-wrong-lane.git
   cd Detect-the-vehicles-in-wrong-lane
   ```

### 2. Create a Virtual Environment
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

### 3. Update the package list and install pip
   ```bash
   sudo apt update
   sudo apt install python3-pip -y
   pip install -U pip
   ```

### 4. Install Required Packages
   Install the Ultralytics package with optional dependencies:
   ```bash
   pip install ultralytics[export]
   ```

### 5. Prepare the Dataset
   Place your training data in the `data/` folder, structured according to YOLOv8 format.  

## Usage
### 1. Train the YOLOv8n Model
   ```bash
   yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50
   ```

### 2. Apply Post-Training Quantization
   ```bash
   python quantize.py --weights weights/yolov8n.pt --quantization ptq
   ```

### 3. Perform Quantization Aware Training
   ```bash
   yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=10 qat=True
   ```

### 4. Run Detection on a Video Stream
   ```bash
   yolo task=detect mode=predict model=yolov8n-quantized.pt source=video.mp4
   ```

### 5. Track and Count Vehicles
   ```bash
   python object_counter.py --video video.mp4 --weights weights/yolov8n-quantized.pt
   ```

## Model Optimization
1. **Post-Training Quantization (PTQ):** Compresses the model by reducing weight precision, improving inference speed with minimal accuracy impact.  
2. **Quantization Aware Training (QAT):** Simulates quantization effects during training for better accuracy in quantized models.  

## Results
- **Performance:**  
   - Model size and inference time reduced using PTQ and QAT.  
   - Accuracy remains comparable to the original YOLOv8n model.  

- **Detection Accuracy:**  
   - Precision: 97%  
   - Recall: 94%  

- **Inference Speed:**  
   - ~40 FPS on GPU.  
   - ~12 FPS on CPU.  

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request to propose changes or improvements.  

---

Nếu bạn cần điều chỉnh thêm thông tin nào, hãy cho mình biết nhé! 😊
