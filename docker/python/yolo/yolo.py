import glob
import cv2
import numpy as np
from more_itertools import peekable
from ultralytics import YOLO
from matplotlib import pyplot as plt
import datetime
import csv
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


def hist_compare(img1, img2, threshold):
    '''
    それぞれの画像をHSV空間に変換後、H(色相)のヒストグラムを比較
    thresholdより大きければ同一物体としてTrueを返す
    '''
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    
    hist1 = cv2.normalize(hist1, None, 0.0, 1.0, cv2.NORM_MINMAX)
    hist2 = cv2.normalize(hist2, None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    res = cv2.compareHist(hist1, hist2, 0)
    if res > threshold:
        return True
    else: return False
    

# start = time.time()
affine_matrix = np.array([[ 1.15775321e+00, 2.06036561e-02, -8.65530736e+01],
                        [-3.59868529e-02, 1.16843440e+00, -4.39524932e+01]])

# path = '/home/srv-admin/images/items*/*/*.jpg'
path = '/home/srv-admin/images/items1/1313/*.jpg'
file_list = peekable(sorted(glob.iglob(path)))

# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# video = cv2.VideoWriter('log.mp4',fourcc, 30.3, (640, 480))

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
            # cv2.drawContours(img_v_color, contours, -1, (0,0,255), 2)
            # cv2.imshow('img', img_v_color)
            # cv2.waitKey(1)
            
            points = []
            for cnt in contours:
                M = cv2.moments(cnt)
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                points.append((cx, cy))
            
            pred = model.predict(img_v_color, classes=[25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79])
            frame = pred[0].plot()
            bboxes = pred[0].boxes.xyxy.cpu().numpy()
            classes = pred[0].boxes.cls.cpu().numpy()
            
            # mask = cv2.subtract(erode_t, erode_v)
            # mask_inv = cv2.bitwise_not(mask)
            # back = cv2.bitwise_and(frame, frame, mask_inv)
            # cut = cv2.bitwise_and(mask, mask, mask)
            # cut = cv2.cvtColor(cut, cv2.COLOR_GRAY2BGR)
            # paste = cv2.add(back, cut)

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
                            data = [[datetime.datetime.now(), "table", int(cls), list(bbox)]]
                            writer = csv.writer(f)
                            writer.writerows(data)
                            
                            xmin, ymin, xmax, ymax = map(int, bbox[:4])
                            crop = img_v_color[ymin:ymax, xmin:xmax]
                            cv2.imwrite('../src/public/images/{}.png'.format(int(cls)), crop)
                            
                            overview = img_v_color.copy()
                            cv2.rectangle(overview, (xmin,ymin), (xmax,ymax), (0, 0, 255), thickness=5)
                            cv2.imwrite('../src/public/images/detail_{}.png'.format(int(cls)), overview)
                            break

            if (feature_compare(b_img_v, img_v)<12):
                bg_v = img_v.copy()
                # print('更新')
            else: b_img_v = img_v.copy()
            
            # video.write(paste)

    except StopIteration:
        break


# video.release()
# end = time.time()
# print(end-start)

