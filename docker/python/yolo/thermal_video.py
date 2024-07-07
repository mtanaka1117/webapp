import glob
import cv2
import numpy as np
from more_itertools import peekable
from ultralytics import YOLO
import datetime
import csv
import os
import shutil
# import time

def feature_compare(img1, img2):
    '''
    画像の特徴量を比較する関数
    '''
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    sift = cv2.ORB_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = bf.match(des1, des2)
    dist = [m.distance for m in matches]
    ret = sum(dist) / len(dist)
    return ret


ksize = 29
def delete_shade(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(img, (ksize, ksize))
    rij = img/(blur+0.0000001)
    index_1 = np.where(rij >= 0.93)
    index_0 = np.where(rij < 0.93)
    rij[index_0] = 0
    rij[index_1] = 255
    return rij


path = '/images/data/0620/table2/*_T.jpg'

file_list = peekable(sorted(glob.iglob(path)))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('thermal_video0620.mp4',fourcc, 30.3, (640, 512), isColor=True)


# model = YOLO("yolov8x-seg.pt")

for i in file_list:
    try:
        img_t = cv2.imread(next(file_list))
        # img_t = 255.0*(img_t - img_t.min())/(img_t.max() - img_t.min())
        img_t = img_t.astype(np.uint8)
        # pred = model.predict(img_t)
        # frame = pred[0].plot()
        
        video.write(img_t)

    except StopIteration:
        break

video.release()
