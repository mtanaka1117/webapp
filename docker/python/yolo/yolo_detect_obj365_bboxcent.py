import glob
import cv2
import numpy as np
from more_itertools import peekable
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
    _, des1 = sift.detectAndCompute(img1, None)
    _, des2 = sift.detectAndCompute(img2, None)

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


# 12-17
# affine_matrix = np.array([[ 1.14911765e+00,  1.08790876e-01, -6.27862149e+01],
#                         [-2.59704731e-02,  1.18778124e+00, -4.43766200e+01]])


# kishino_1223
affine_matrix = np.array([[ 1.53104533e+00, -1.88203939e-02,  3.54805456e+00],
                        [-3.01498812e-03,  1.49792624e+00,  2.71967074e+01]])

#kishino_0113
# affine_matrix = np.array([[ 1.62759602e+00,  3.96586060e-01, -3.93778094e+01],
#                         [-8.42105263e-03,  1.65052632e+00,  1.49515789e+01]])

# kishino_0116
# affine_matrix = np.array([[1.54433829e+00, 1.00447774e-03, 6.78595218e+00],
#                         [1.80839216e-03, 1.55654705e+00, 1.09340073e+01]])

# kishino_0119
# affine_matrix = np.array([[ 1.59084405, -0.053231,    4.39110465],
#                         [ 0.04086998,  1.43773675, 28.14188174]])


root_path = "/images/kishino/20241223_2031/"
pattern = "*.jpg"

file_list = []
for dirpath, _ , _ in os.walk(root_path):
    file_list.extend(glob.glob(os.path.join(dirpath, pattern)))
file_list = peekable(sorted(file_list))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('./video/kishino_obj365_1223_2031.mp4', fourcc, 30, (1065, 850), isColor=True)

if '_V' in file_list.peek():
    bg_v = cv2.imread(next(file_list))
    bg_t = cv2.imread(next(file_list))
else:
    next(file_list)
    bg_v = cv2.imread(next(file_list))
    bg_t = cv2.imread(next(file_list))

# if '_T' in file_list.peek():
#     bg_t = cv2.imread(next(file_list))
#     bg_v = cv2.imread(next(file_list))
# else:
#     next(file_list)
#     bg_t = cv2.imread(next(file_list))
#     bg_v = cv2.imread(next(file_list))


# 1つ前のフレーム
b_img_v = bg_v.copy()
kernel = np.ones((5,5), np.uint8)

model = YOLOWorld("yolov8s-world.pt")
model.set_classes(["Person", "Glasses", "Bottle", "Cup", "Handbag",
        "Book", "Umbrella", "Watch", "Pen/Pencil", "Cell Phone",
        "Clock", "Head Phone", "Remote", "Scissors",
        "Folder", "earphone", "Mask", "Tissues", "Wallet",
        "Tablet", "Key", "Stapler", "Eraser", "Lipstick"])
flag_hand = 0

shutil.rmtree('../results/thing2vec_1223_2031_thumbnails/')
shutil.rmtree('../results/thing2vec_1223_2031_detail/')
shutil.rmtree('../results/thing2vec_1223_2031_mask/')
os.remove('./log/thing2vec_kishino_1223_2031.csv')


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
            _, img_th_t = cv2.threshold(affined_t,25,255,cv2.THRESH_BINARY)
            erode_t = cv2.dilate(img_th_t,kernel,3)
            # erode_t = cv2.erode(dilate_t, kernel, 3)

            touch_region = cv2.subtract(erode_t, erode_v)
            # touch_region = cv2.bitwise_and(touch_region, table_mask)
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
            bboxes_xyxy = pred[0].boxes.xyxy.cpu().numpy()
            bboxes_xywh = pred[0].boxes.xywh.cpu().numpy() # x_center, y_center, width, height
            classes = pred[0].boxes.cls.cpu().numpy()
            
            
            # 人の手があるとき
            if 0 in classes: flag_hand = 1
            
            polygon = [] # 検知した物体のbboxを格納
            for x1, y1, x2, y2 in np.round(bboxes_xyxy, 1):
                polygon.append(np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]))
            
            
            mask = touch_region
            mask_inv = cv2.bitwise_not(mask)
            back = cv2.bitwise_and(frame, frame, mask_inv)
            cut = cv2.bitwise_and(mask, mask, mask)
            cut = cv2.cvtColor(cut, cv2.COLOR_GRAY2BGR)
            paste = cv2.add(back, cut)
            

            with open("./log/thing2vec_kishino_1223_2031.csv", "a") as f:
                for poly, cls, bbox_xyxy, bbox_xywh in zip(polygon, classes, bboxes_xyxy, bboxes_xywh):
                    is_touch = False
                    
                    for pt in points: # あるpolyに対して熱痕跡の重心があるかどうか
                        if cv2.pointPolygonTest(poly, pt, False) >= 0 and cls != 0:
                            is_touch = True
                    
                    if cls != 0:
                        time, dow = extract_datetime_from_filename(i)
                        data = [[time, dow, "home", int(cls), list(map(int, bbox_xyxy)), list(map(int, bbox_xywh)), is_touch]]
                        writer = csv.writer(f)
                        writer.writerows(data)
                        
                        path = '../results/thing2vec_1223_2031_thumbnails/{}'.format(int(cls))
                        if not os.path.exists(path): os.makedirs(path)
                        path = '../results/thing2vec_1223_2031_detail/{}'.format(int(cls))
                        if not os.path.exists(path): os.makedirs(path)
                        path = '../results/thing2vec_1223_2031_mask/{}'.format(int(cls))
                        if not os.path.exists(path): os.makedirs(path)
                        
                        xmin, ymin, xmax, ymax = map(int, bbox_xyxy[:4])
                        crop = img_v_color[ymin:ymax, xmin:xmax]
                        cv2.imwrite('../results/thing2vec_1223_2031_thumbnails/{}/{}.jpg'.format(int(cls),time), crop)
                        
                        overview = img_v_color.copy()
                        cv2.rectangle(overview, (xmin,ymin), (xmax,ymax), (0, 0, 255), thickness=5)
                        cv2.imwrite('../results/thing2vec_1223_2031_detail/{}/detail_{}.jpg'.format(int(cls),time), overview)
                        
                        mask = cv2.bitwise_and(erode_v, erode_t)
                        mask_inv = cv2.bitwise_not(mask)[ymin:ymax, xmin:xmax]
                        cv2.imwrite('../results/thing2vec_1223_2031_mask/{}/{}.jpg'.format(int(cls),time), mask_inv)
                        break

            if feature_compare(b_img_v, img_v)<12:
            # if (0 not in classes and feature_compare(b_img_v, img_v)<12): #手がない時に背景更新
                bg_v = img_v.copy()
                b_img_v = img_v.copy()
                # bg_t = img_t.copy()
                
            else: b_img_v = img_v.copy()
            
            video.write(paste)

    except StopIteration:
        break

video.release()

