
# Detect Vehicle in Wrong Lane

This project leverages **computer vision** and **deep learning** techniques to detect vehicles traveling in the wrong lane. A lightweight and efficient detection model based on **YOLOv11n** is employed, with further optimization using **Post-Training Quantization (PTQ)**. The system supports real-time object detection, tracking, and violation analysis, and is designed to run smoothly on low-power devices such as the **Raspberry Pi 5**.

## Table of Contents
- [Overview](#overview)  
- [Features](#features)  
- [Technologies Used](#technologies-used)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Optimization](#model-optimization)  
- [Results](#results)  
- [Demo](#demo) 
- [Summary](#summary)
- [Contributing](#contributing)  

## Overview

This project aims to develop an intelligent system that can:
1. **Train** a custom vehicle detection model using **YOLOv11n**.
2. **Optimize** the model with **Post-Training Quantization (PTQ)** to reduce size and improve inference time.
3. **Detect and track** vehicles using **YOLOv11n** and **ByteTrack**, then compare their positions against predefined lane boundaries to identify violations.

## Features

- ✅ **Real-Time Detection:** Accurately detects wrong-lane driving behavior.
- 🚀 **Quantized Model:** Optimized using PTQ for improved performance and lower resource usage.
- 🎯 **Accurate Classification:** Supports motorbikes, cars, trucks, and buses.
- 💻 **Edge Deployment:** Efficiently runs on embedded systems like Raspberry Pi 5.

## Technologies Used

- **YOLOv11n** – Object detection architecture.
- **Ultralytics** – YOLO training/export toolkit.
- **OpenVINO** – For INT8 model inference optimization.
- **Python**, **OpenCV** – For image processing and application logic.
- **ByteTrack** – Multi-object tracking framework.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/DangUIT/Detect-the-vehicles-in-wrong-lane.git
cd Detect-the-vehicles-in-wrong-lane
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

### 3. Install Dependencies
```bash
sudo apt update
sudo apt install python3-pip -y
pip install -U pip
pip install ultralytics[export]
```



## Usage

### 1. Navigate to source directory
```bash
cd ./src
```

### 2. Run Detection

#### On **Windows**:
```bash
.\run_pvd.bat
.\run_bentre.bat
```

#### On **Raspberry Pi 5**:
```bash
./run_pvd.sh
./run_bentre.sh
```

## Model Optimization

- **Post-Training Quantization (PTQ):**  
  The model is converted to **INT8** format using **OpenVINO**, which significantly improves inference speed and reduces model size. This allows for smooth and efficient deployment on embedded platforms such as Raspberry Pi, with minimal accuracy degradation.

## Results

### Detection Accuracy

| Model                   | Precision | Recall | mAP50 |
|------------------------|-----------|--------|--------|
| YOLOv11n_224x224        | 0.851     | 0.770  | 0.859  |
| YOLOv11n_INT8_224x224   | 0.848     | 0.765  | 0.855  |
| YOLOv11n_416x416        | 0.902     | 0.886  | 0.948  |
| YOLOv11n_INT8_416x416   | 0.884     | 0.882  | 0.943  |
| YOLOv11n_640x640        | 0.910     | 0.914  | 0.962  |
| YOLOv11n_INT8_640x640   | 0.905     | 0.911  | 0.962  |

### Inference Performance on Raspberry Pi 5

| Model                   | FPS   | Inference Time |
|------------------------|-------|----------------|
| YOLOv11n_224x224        | 14.85 | 60.69 ms       |
| YOLOv11n_INT8_224x224   | 30.17 | 23.60 ms       |
| YOLOv11n_640x640        | 3.79  | 250.42 ms      |
| YOLOv11n_INT8_640x640   | 8.68  | 98.57 ms       |

📁 Output videos are saved in: `result/video`  
📄 Benchmark logs are saved in: `result/benchmark`

## Demo

![YouTube Logo](https://img.icons8.com/color/12/000000/youtube-play.png) [Wrong-lane detection – Ben Tre](https://youtu.be/1P9afBQDIDM)  
![YouTube Logo](https://img.icons8.com/color/12/000000/youtube-play.png)[Wrong-lane detection – Pham Van Dong](https://youtu.be/WX-ibKRnSQ0)

## Summary

This project delivers an effective and lightweight traffic monitoring system capable of detecting wrong-lane vehicle movement. It is specifically designed for **real-time deployment** on **embedded devices** like Raspberry Pi.

Key highlights:

🚗 **Vehicle Type Support:** Motorbike, car, truck, bus  
🛣️ **Violation Analysis:** Detects vehicles crossing into incorrect lanes  
🎯 **Reliable Tracking:** Powered by ByteTrack for object association  
⚡ **Performance Boost:** Quantized model offers 2–3× speedup with reduced CPU load  

This solution demonstrates strong potential for **smart traffic management** and urban safety applications.

💵 *If you need to be supported, please contact: trandanganninh1@gmail.com (paid request)*
## Contributing

Contributions are welcome!  
Feel free to fork this repository and submit a pull request with improvements or new features.
