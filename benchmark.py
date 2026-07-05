import cv2
import numpy as np
import onnxruntime as ort
from time import perf_counter
import os


session = ort.InferenceSession("models/best.onnx")
image = cv2.imread("data/imagetest/image1.jpeg")
image_resized = cv2.resize(image, (640, 640))
image_normalized = image_resized / 255.0
image_tensor = image_normalized.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
input_name = session.get_inputs()[0].name

session.run(None, {input_name: image_tensor})

runs = 100
start = perf_counter()
for _ in range(runs):
    session.run(None, {input_name: image_tensor})
end = perf_counter()

total_ms = (end - start) * 1000
avg_ms = total_ms / runs
fps = 1000 / avg_ms
model_size_mb = os.path.getsize("models/best.onnx") / (1024 * 1024)

print(f"\n--- FP32 Benchmark ---")
print(f"Average inference time: {avg_ms:.2f} ms")
print(f"Throughput: {fps:.1f} FPS")
print(f"Model size: {model_size_mb:.1f} MB")




# INT8 benchmark
session_int8 = ort.InferenceSession("models/best_int8.onnx")
session_int8.run(None, {input_name: image_tensor})  # warmup

start = perf_counter()
for _ in range(runs):
    session_int8.run(None, {input_name: image_tensor})
end = perf_counter()

total_ms_int8 = (end - start) * 1000
avg_ms_int8 = total_ms_int8 / runs
fps_int8 = 1000 / avg_ms_int8
model_size_int8 = os.path.getsize("models/best_int8.onnx") / (1024 * 1024)

print(f"\n--- INT8 Benchmark ---")
print(f"Average inference time: {avg_ms_int8:.2f} ms")
print(f"Throughput: {fps_int8:.1f} FPS")
print(f"Model size: {model_size_int8:.1f} MB")

print(f"\n--- Comparison ---")
print(f"Size reduction: {model_size_mb/model_size_int8:.1f}x smaller")
print(f"Speed improvement: {avg_ms/avg_ms_int8:.1f}x faster")
