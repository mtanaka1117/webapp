import glob
import cv2
import numpy as np
from more_itertools import peekable
from ultralytics import YOLO
import datetime
import csv
import os
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
    

affine_matrix = np.array([[ 1.15775321e+00, 2.06036561e-02, -8.65530736e+01],
                        [-3.59868529e-02, 1.16843440e+00, -4.39524932e+01]])

path = '/images/items*/*/*.jpg'
# path = '/images/items1/1313/*.jpg'

def yolo(path=path):
    file_list = peekable(sorted(glob.iglob(path)))

    if '_V' in file_list.peek():
        bg_v = cv2.imread(next(file_list), 0)
        bg_t = cv2.imread(next(file_list), 0)
    else:
        next(file_list)
        bg_v = cv2.imread(next(file_list), 0)
        bg_t = cv2.imread(next(file_list), 0)
        
    b_img_v = bg_v.copy()
    kernel = np.ones((5,5),np.uint8)

    model = YOLO("yolov8x.pt")

    classes = list(range(1, 80))

    for i in file_list:
        try:
            if '_V' in i and '_T' in file_list.peek():
                img_v = cv2.imread(i, 0)
                img_v_color = cv2.imread(i)
                diff_v = cv2.absdiff(img_v, bg_v)
                _, img_th_v = cv2.threshold(diff_v,30,255,cv2.THRESH_BINARY)
                dilate_v = cv2.dilate(img_th_v,kernel,iterations=5)
                erode_v = cv2.erode(dilate_v, kernel, 2)

                img_t = cv2.imread(next(file_list), 0)
                diff_t = cv2.absdiff(img_t, bg_t)
                affined_t = cv2.warpAffine(diff_t, affine_matrix, (img_v.shape[1], img_v.shape[0]))
                _, img_th_t = cv2.threshold(affined_t,12,255,cv2.THRESH_BINARY)
                dilate_t = cv2.dilate(img_th_t,kernel,3)
                erode_t = cv2.erode(dilate_t, kernel, 2)

                touch_region = cv2.subtract(erode_t, erode_v)
                contours, _ = cv2.findContours(touch_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = list(filter(lambda x: cv2.contourArea(x) > 80, contours))
                
                points = []
                for cnt in contours:
                    M = cv2.moments(cnt)
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    points.append((cx, cy))
                
                pred = model.predict(img_v_color, classes=[25, 26, 28, 39, 41, 64, 65, 67, 73, 74, 76])
                bboxes = pred[0].boxes.xyxy.cpu().numpy()
                classes = pred[0].boxes.cls.cpu().numpy()
                
                polygon = []
                for x1, y1, x2, y2 in bboxes:
                    polygon.append(np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]))

                with open("result.csv", "a") as f:
                    for poly, cls, bbox in zip(polygon, classes, bboxes):
                        # data = [[datetime.datetime.now(), "table", cls, list(bbox)]]
                        # writer = csv.writer(f)
                        # writer.writerows(data)
                        for pt in points:
                            if cv2.pointPolygonTest(poly, pt, False) >= 0:
                                time = datetime.datetime.now()
                                data = [[time, "table", int(cls), list(bbox)]]
                                writer = csv.writer(f)
                                writer.writerows(data)
                                
                                path = '../results/thumbnails/{}'.format(int(cls))
                                if not os.path.exists(path): os.mkdir(path)
                                path = '../results/detail/{}'.format(int(cls))
                                if not os.path.exists(path): os.mkdir(path)
                                
                                xmin, ymin, xmax, ymax = map(int, bbox[:4])
                                crop = img_v_color[ymin:ymax, xmin:xmax]
                                cv2.imwrite('../results/thumbnails/{}/{}.png'.format(int(cls),time), crop)
                                
                                overview = img_v_color.copy()
                                cv2.rectangle(overview, (xmin,ymin), (xmax,ymax), (0, 0, 255), thickness=5)
                                cv2.imwrite('../results/detail/{}/detail_{}.png'.format(int(cls),time), overview)
                                break

                if (feature_compare(b_img_v, img_v)<12):
                    bg_v = img_v.copy()
                    # print('更新')
                else: b_img_v = img_v.copy()
                
        except StopIteration:
            break
        
if __name__ == '__main__':
    yolo()