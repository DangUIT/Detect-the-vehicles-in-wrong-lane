from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model="D:/Project/train/imgsz_640/best.pt",
          data="D:/Project/train/dataset_1920_1080/data.yaml",
          imgsz=640,
          format="")

