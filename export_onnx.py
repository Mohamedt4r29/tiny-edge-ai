from ultralytics import YOLO
model = YOLO("models/best.pt")
model.export(format="onnx")
print("Export done. Check models/ for best.onnx")
