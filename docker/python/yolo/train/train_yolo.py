from ultralytics import YOLO

# Load a model
model = YOLO("../yolov8x.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="Objects365.yaml", epochs=100, imgsz=640)
