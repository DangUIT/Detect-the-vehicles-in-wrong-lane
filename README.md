

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
- [Demo](#demo) 
- [Contributing](#contributing)  


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
- **Scalable:** Adaptable to various traffic scenarios and datasets.  

## Technologies Used
- **YOLOv8n** – for model training and object detection.  
- **Ultralytics** – Python package for YOLOv8.  
- **Python** – main programming language.  
- **OpenCV** – for image and video processing.  


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


## Usage


### 1. Run Video
   ```bash
   cd .\src\     
   python .\main.py --video ..\video\Video\pvd_front.mp4
   or
   python3 .\main.py --video ..\video\Video\pvd_front.mp4

   ```


## Model Optimization
1. **Post-Training Quantization (PTQ):** Compresses the model by reducing weight precision, improving inference speed with minimal accuracy impact.  
2. **Quantization Aware Training (QAT):** Simulates quantization effects during training for better accuracy in quantized models.  

## Results
- Video result will be stored in result/video
- **Performance:**  
   - Model size and inference time reduced using PTQ and QAT.  
   - Accuracy remains comparable to the original YOLOv8n model.  

- **Detection Accuracy:**  
   - Precision: 93.5%  
   - Recall: 89.1%  

- **Inference Speed in Raspberry Pi 5:**  
   - Initial Model: FPS = 4.83  
   - Static Post Training Quantization: FPS = 12.73.  

## Demo
- [Detecting vehicle in wrong lane](https://youtu.be/XSQ5pRy6Tq0)
## Contributing
Contributions are welcome! Please fork the repository and submit a pull request to propose changes or improvements.  

---
