#!/bin/bash

python3 main.py --video ../video/Video/pvd_front.mp4 --model ../train/imgsz_224/best.pt --size 224 224
sleep 1

python3 main.py --video ../video/Video/pvd_front.mp4 --model ../train/imgsz_224/best_int8_openvino_model/ --size 224 224
sleep 1

python3 main.py --video ../video/Video/pvd_front.mp4 --model ../train/imgsz_416/best.pt --size 416 416
sleep 1

python3 main.py --video ../video/Video/pvd_front.mp4 --model ../train/imgsz_416/best_int8_openvino_model/ --size 416 416
sleep 1

python3 main.py --video ../video/Video/pvd_front.mp4 --model ../train/imgsz_640/best.pt --size 640 640
sleep 1

python3 main.py --video ../video/Video/pvd_front.mp4 --model ../train/imgsz_640/best_int8_openvino_model/ --size 640 640
