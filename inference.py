import cv2
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession("models/best.onnx")

# Read image
image = cv2.imread("data/imagetest/image1.jpeg")

# Preprocess
image_resized = cv2.resize(image, (640, 640))
image_normalized = image_resized / 255.0
image_tensor = image_normalized.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)

# Inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: image_tensor})

print("Output shape:", outputs[0].shape)

# -------------------------
# POSTPROCESSING STARTS
# -------------------------

# Remove batch dimension + transpose
prediction = outputs[0][0].T   # (8400, 8)

print("Prediction shape:", prediction.shape)

# Split boxes and class scores
boxes = prediction[:, :4]
scores = prediction[:, 4:]

# IMPORTANT: apply sigmoid (fix for raw outputs)
#scores = 1 / (1 + np.exp(-scores))
# Get best class per box
class_ids = np.argmax(scores, axis=1)

# Get confidence (max class score)
confidences = np.max(scores, axis=1)

# Threshold filter
threshold = 0.5
mask = confidences > threshold

# Apply mask
boxes = boxes[mask]
class_ids = class_ids[mask]
confidences = confidences[mask]


boxes_tl = boxes.copy()
boxes_tl[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
boxes_tl[:, 1] = boxes[:, 1] - boxes[:, 3] / 2

indices = cv2.dnn.NMSBoxes(boxes_tl.tolist(), confidences.tolist(), 0.5, 0.45)

class_names = ["cattle", "chicken", "duck", "goat"]
print(f"\nDetections after NMS: {len(indices)}")
for i in indices:
    print(f"  {class_names[class_ids[i]]}: {confidences[i]:.2f} confidence")

#Scale boxes back to 640x640
#boxes *= 640

# -------------------------
# PRINT RESULTS
# -------------------------


