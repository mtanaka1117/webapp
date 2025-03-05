from ultralytics import YOLOWorld
import cv2

path = '/images/kishino/20250120_1958/20250120_1955/20250120_195526739_V.jpg'
# path = '/images/kishino/20241223_1912/20241223_1906/20241223_190600846_V.jpg'

model = YOLOWorld("yolov8x-worldv2.pt")
model.set_classes(["Person", "Glasses", "Bottle", "Cup", "Handbag",
        "Book", "Umbrella", "Watch", "Pen/Pencil", "Cell Phone",
        "Clock", "Head Phone", "Remote", "Scissors",
        "Folder", "earphone", "Mask", "Tissues", "Wallet",
        "Tablet", "Key", "Stapler", "Eraser", "Lipstick"])

img = cv2.imread(path)

pred = model.predict(img)
frame = pred[0].plot()

cv2.imwrite('test.jpg', frame)
