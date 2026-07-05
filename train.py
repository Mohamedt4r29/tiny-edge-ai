from ultralytics import YOLO
model = YOLO("yolov8n.pt")

model.train(
    data="data/livestock/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="results",
    name="livestock_v1"
)
