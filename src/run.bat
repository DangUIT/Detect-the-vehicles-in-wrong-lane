python main.py --video ..\video\Video\bentre.mp4 --model ..\train\imgsz_224\best.pt --size 224 224
timeout /t 1 >nul

python main.py --video ..\video\Video\bentre.mp4 --model ..\train\imgsz_224\best_int8_openvino_model\ --size 224 224
timeout /t 1 >nul

python main.py --video ..\video\Video\bentre.mp4 --model ..\train\imgsz_416\best.pt --size 416 416
timeout /t 1 >nul

python main.py --video ..\video\Video\bentre.mp4 --model ..\train\imgsz_416\best_int8_openvino_model\ --size 416 416
timeout /t 1 >nul

python main.py --video ..\video\Video\bentre.mp4 --model ..\train\imgsz_640\best.pt --size 640 640
timeout /t 1 >nul

python main.py --video ..\video\Video\bentre.mp4 --model ..\train\imgsz_640\best_int8_openvino_model\ --size 640 640
timeout /t 1 >nul