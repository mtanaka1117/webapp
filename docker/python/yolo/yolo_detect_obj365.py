import glob
import cv2
import numpy as np
from more_itertools import peekable
from ultralytics import YOLO
import csv
import os
import shutil
from pathlib import Path
import re
import datetime
from ultralytics import YOLOWorld


def feature_compare(img1, img2):
    '''
    画像の特徴量を比較する関数
    '''
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    sift = cv2.ORB_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is not None and des2 is not None:
        matches = bf.match(des1, des2)
        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)
        return ret
    else:
        return 1000
    
    
def extract_datetime_from_filename(filename):
    dow_dict = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }
    
    filename = Path(filename).name
    pattern = r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})"
    match = re.search(pattern, filename)

    if match:
        year, month, day, hour, minute, second = match.groups()
        dt = datetime.datetime(int(year), int(month), int(day))
        dow = dow_dict[dt.weekday()]
        return f"{year}-{month}-{day} {hour}:{minute}:{second}", dow
    else:
        return None, None


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


# table1
affine_matrix = np.array([[ 1.18039980e+00,  7.09369517e-02, -8.98547621e+01],
                        [-3.24900007e-02,  1.16626013e+00, -3.75450685e+01]])

# table2
# affine_matrix = np.array([[ 1.17217602e+00,  8.45247542e-02, -9.23932162e+01],
#                         [-7.13793184e-02,  1.16531538e+00, -2.59984213e+01]])

# table3
# affine_matrix = np.array([[ 1.17012871e+00,  1.08532231e-01, -8.49767956e+01],
#                         [-3.78228456e-02,  1.17486284e+00, -3.45068805e+01]])

# path = '/images/yolo+1/20240112_1450/*jpg'
# path = '/images/yolo+1/*/*jpg'
path = '/images/data/0731/table1/*jpg'

file_list = peekable(sorted(glob.iglob(path)))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('test_table1_obj365.mp4',fourcc, 30.3, (640, 480), isColor=True)

# if '_V' in file_list.peek():
#     bg_v = cv2.imread(next(file_list))
#     bg_t = cv2.imread(next(file_list))
# else:
#     next(file_list)
#     bg_v = cv2.imread(next(file_list))
#     bg_t = cv2.imread(next(file_list))

if '_T' in file_list.peek():
    bg_t = cv2.imread(next(file_list))
    bg_v = cv2.imread(next(file_list))
else:
    next(file_list)
    bg_t = cv2.imread(next(file_list))
    bg_v = cv2.imread(next(file_list))

# 1つ前のフレーム
b_img_v = bg_v.copy()

# 1つ前の背景
bg_before = delete_shade(bg_v)
kernel = np.ones((5,5), np.uint8)

# 初期背景（何も置かれていない）
# bg_first_v = bg_v.copy()
# bg_first_v_gray = cv2.cvtColor(bg_first_v, cv2.COLOR_BGR2GRAY)
bg_first_v = cv2.imread('/images/data/0731/test_table1_V.jpg')
bg_first_v_gray = cv2.cvtColor(bg_first_v, cv2.COLOR_BGR2GRAY)
_, table_mask = cv2.threshold(bg_first_v_gray, 80, 255, cv2.THRESH_BINARY)


# model = YOLO("yolov8x-oiv7.pt")
model = YOLOWorld("yolov8s-world.pt")
model.set_classes(["Person","Glasses", "Bottle", "Cup", "Handbag/Satchel", "Book",
        "Umbrella", "Watch", "Pen/Pencil", "Cell Phone",
        "Laptop", "Clock", "Keyboard", "Mouse", "Head Phone", "Remote",
        "Scissors", "Folder", "earphone", "Mask", "Tissue", "Wallet/Purse",
        "Tablet", "Key", "CD", "Stapler", "Eraser", "Lipstick"])
flag_hand = 0
color_g = {}


# shutil.rmtree('../results/thing2vec_table1_thumbnails/')
# shutil.rmtree('../results/thing2vec_table1_detail/')
# shutil.rmtree('../results/thing2vec_table1_mask/')
# os.remove('./log/thing2vec_table1.csv')


for i in file_list:
    try:
        if '_V' in i and '_T' in file_list.peek():
            img_v = cv2.imread(i)
            img_v_color = cv2.imread(i)
            diff_v = cv2.absdiff(img_v, bg_v)
            diff_v = cv2.cvtColor(diff_v, cv2.COLOR_BGR2GRAY)
            _, img_th_v = cv2.threshold(diff_v,12,255,cv2.THRESH_BINARY)
            dilate_v = cv2.dilate(img_th_v,kernel,3)
            erode_v = cv2.erode(dilate_v, kernel, 3)

            img_t = cv2.imread(next(file_list))
            diff_t = cv2.absdiff(img_t, bg_t)
            diff_t = cv2.cvtColor(diff_t, cv2.COLOR_BGR2GRAY)
            affined_t = cv2.warpAffine(diff_t, affine_matrix, (img_v.shape[1], img_v.shape[0]))
            _, img_th_t = cv2.threshold(affined_t,10,255,cv2.THRESH_BINARY)
            dilate_t = cv2.dilate(img_th_t,kernel,3)
            erode_t = cv2.erode(dilate_t, kernel, 3)

            touch_region = cv2.subtract(erode_t, erode_v)
            touch_region = cv2.bitwise_and(touch_region, table_mask)
            contours, _ = cv2.findContours(touch_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = list(filter(lambda x: cv2.contourArea(x) > 80, contours))
            
            # 熱痕跡の重心
            points = []
            for cnt in contours:
                M = cv2.moments(cnt)
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                points.append((cx, cy))

            # YOLO
            pred = model.predict(img_v_color)
            frame = pred[0].plot()
            bboxes = pred[0].boxes.xyxy.cpu().numpy()
            classes = pred[0].boxes.cls.cpu().numpy()
            
            
            # 人の手があるとき
            if 0 in classes: flag_hand = 1
            
            polygon = [] # 検知した物体のbboxを格納
            for x1, y1, x2, y2 in np.round(bboxes, 1):
                polygon.append(np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]))
            
            
            mask = touch_region
            mask_inv = cv2.bitwise_not(mask)
            back = cv2.bitwise_and(frame, frame, mask_inv)
            cut = cv2.bitwise_and(mask, mask, mask)
            cut = cv2.cvtColor(cut, cv2.COLOR_GRAY2BGR)
            paste = cv2.add(back, cut)
            

            with open("./log/thing2vec_table1.csv", "a") as f:
                for poly, cls, bbox in zip(polygon, classes, bboxes):
                    is_touch = False
                    
                    for pt in points: # あるpolyに対して熱痕跡の重心があるかどうか
                        if cv2.pointPolygonTest(poly, pt, False) >= 0 and cls != 0:
                            is_touch = True
                    
                    time, dow = extract_datetime_from_filename(i)
                    data = [[time, dow, "table1", int(cls), list(bbox), is_touch]]
                    writer = csv.writer(f)
                    writer.writerows(data)
                    
                    path = '../results/thing2vec_table1_thumbnails/{}'.format(int(cls))
                    if not os.path.exists(path): os.makedirs(path)
                    path = '../results/thing2vec_table1_detail/{}'.format(int(cls))
                    if not os.path.exists(path): os.makedirs(path)
                    path = '../results/thing2vec_table1_mask/{}'.format(int(cls))
                    if not os.path.exists(path): os.makedirs(path)
                    
                    xmin, ymin, xmax, ymax = map(int, bbox[:4])
                    crop = img_v_color[ymin:ymax, xmin:xmax]
                    cv2.imwrite('../results/thing2vec_table1_thumbnails/{}/{}.jpg'.format(int(cls),time), crop)
                    
                    overview = img_v_color.copy()
                    cv2.rectangle(overview, (xmin,ymin), (xmax,ymax), (0, 0, 255), thickness=5)
                    cv2.imwrite('../results/thing2vec_table1_detail/{}/detail_{}.jpg'.format(int(cls),time), overview)
                    
                    mask = cv2.bitwise_and(erode_v, erode_t)
                    mask_inv = cv2.bitwise_not(mask)[ymin:ymax, xmin:xmax]
                    cv2.imwrite('../results/thing2vec_table1_mask/{}/{}.jpg'.format(int(cls),time), mask_inv)
                    break


            # if (feature_compare(b_img_v, img_v)<12):
            if (0 not in classes and feature_compare(b_img_v, img_v)<12): #手がない時に背景更新
                bg_v = img_v.copy()
                b_img_v = img_v.copy()
                # bg_t = img_t.copy()
                
            else: b_img_v = img_v.copy()
            
            video.write(paste)

    except StopIteration:
        break

video.release()

