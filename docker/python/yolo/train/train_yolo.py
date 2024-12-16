from ultralytics import YOLO
import ultralytics.data.build as build
from dataloader import YOLOWeightedDataset
import numpy as np

# Load a model
model = YOLO("../yolov8x.pt")  # load a pretrained model (recommended for training)

build.YOLODataset = YOLOWeightedDataset
# class_counts = [80332, 6676, 24399, 22608, 6832, 6827, 3271, 2724, 9264, 4131,
#                 4614, 1702, 5925, 5877, 2126, 2257, 2712, 1566, 1414, 459, 2027,
#                 886, 917, 1031, 1139, 506, 263, 248]
# total_count = sum(class_counts)
# class_weights = [total_count / count for count in class_counts]
# print(np.round(class_weights, 1))

# Train the model
results = model.train(data="Objects365.yaml", epochs=50, imgsz=640, pretrained=True, batch=32, device=[0,1])