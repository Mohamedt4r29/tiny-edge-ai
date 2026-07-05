# Tiny Vision Inference Engine

> Edge AI pipeline for livestock detection — PyTorch → ONNX → C++ Runtime

---

## Overview

A complete Edge AI inference pipeline for farm animal detection. The pipeline covers the full embedded AI workflow: train a model on a GPU machine, export to a portable format, quantize for size and speed, and benchmark the tradeoffs. Built to demonstrate the skills relevant to embedded AI deployment on resource-constrained devices.

**Detected classes:** cattle, chicken, duck, goat

---

## Pipeline

<img width="2069" height="3835" alt="edge" src="https://github.com/user-attachments/assets/6636d15c-cb70-4f18-9dbc-7b88b7f52e69" />


## Results

### Training — YOLOv8n, 50 epochs, Google Colab T4

| Class   | mAP50 | Precision | Recall |
|---------|-------|-----------|--------|
| cattle  | 0.862 | 0.876     | 0.781  |
| chicken | 0.837 | 0.824     | 0.761  |
| duck    | 0.801 | 0.784     | 0.738  |
| goat    | 0.820 | 0.836     | 0.774  |
| **all** | **0.830** | **0.830** | **0.764** |

### Benchmark — Intel Core i7-10750H (CPU)

| Metric          | FP32       | INT8       |
|-----------------|------------|------------|
| Model size      | 11.7 MB    | 3.2 MB     |
| Inference time  | 52.88 ms   | 198.98 ms  |
| Throughput      | 18.9 FPS   | 5.0 FPS    |
| Size reduction  | —          | 3.7× smaller |

> **Note on INT8 speed:** Dynamic quantization is slower on CPU because weights are
> dequantized at runtime. On hardware with dedicated INT8 execution units
> (ARM Cortex-M with CMSIS-NN, NXP i.MX RT), INT8 delivers 2–4× speedup over FP32.

---

## Project Structure

```
tiny-edge-ai/
├── train.py            # YOLOv8n training on livestock dataset
├── export_onnx.py      # Export trained weights to ONNX format
├── inference.py        # ONNX Runtime inference + NMS postprocessing
├── benchmark.py        # FP32 vs INT8 speed and size benchmark
├── quantize.py         # INT8 dynamic quantization
├── download_data.py    # Roboflow dataset download
├── models/
│   ├── best.pt         # PyTorch weights (6.0 MB)
│   ├── best.onnx       # FP32 ONNX model (11.7 MB)
│   └── best_int8.onnx  # INT8 quantized model (3.2 MB)
├── data/
│   └── livestock/      # Dataset (cattle, chicken, duck, goat)
├── results/            # Training curves and validation plots
└── cpp_runtime/        # C++ inference (in progress)
```

---

## Concepts Demonstrated

**ML pipeline**
- YOLOv8n training with transfer learning from COCO pretrained weights
- ONNX export and format internals (8400 candidate boxes across 3-scale FPN)
- Postprocessing: transpose → confidence filter → NMS
- INT8 dynamic quantization via ONNX Runtime

**Embedded AI**
- ONNX as a framework-neutral model interchange format
- FP32 vs INT8 tradeoff: accuracy vs size vs speed
- Why INT8 speed gains require dedicated hardware (INT8 ALU)
- Stride formula: mapping grid cell indices to image pixel coordinates
- Memory implications of 3-scale feature pyramid (80×80 + 40×40 + 20×20 = 8400 boxes)

---

## Dataset

**livestock\_detection v4** — Roboflow Universe  
5,434 images | 4 classes | CC BY 4.0 license  
Split: 84% train / 11% val / 5% test

```bibtex
@misc{livestock_detection_dataset,
  title  = {livestock\_detection Dataset},
  author = {livestock},
  year   = {2024},
  url    = {https://universe.roboflow.com/livestock/livestock_detection}
}
```

---

## How to Run

```bash
# Clone and set up
git clone https://github.com/Mohamedt4r29/tiny-edge-ai
cd tiny-edge-ai
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision ultralytics onnx onnxruntime roboflow

# Download dataset
python3 download_data.py

# Train
python3 train.py

# Export to ONNX
python3 export_onnx.py

# Run inference on a test image
python3 inference.py

# Quantize to INT8
python3 quantize.py

# Benchmark FP32 vs INT8
python3 benchmark.py
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Training | PyTorch + Ultralytics YOLOv8 |
| Export | ONNX opset 20 + onnxslim |
| Runtime | ONNX Runtime 1.27 |
| Quantization | onnxruntime.quantization dynamic INT8 |
| Dataset | Roboflow Universe |
| Training hardware | Google Colab Tesla T4 |
| Inference hardware | Intel i7-10750H (CPU) |
