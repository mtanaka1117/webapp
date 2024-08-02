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

    if des1 is not None and des2 is not None:
        matches = bf.match(des1, des2)
        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)
        return ret
    else:
        return 1000


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

# start = time.time()

# table1
affine_matrix = np.array([[ 1.18039980e+00,  7.09369517e-02, -8.98547621e+01],
                        [-3.24900007e-02,  1.16626013e+00, -3.75450685e+01]])

# table2
# affine_matrix = np.array([[ 1.17217602e+00,  8.45247542e-02, -9.23932162e+01],
#                         [-7.13793184e-02,  1.16531538e+00, -2.59984213e+01]])

# table3
# affine_matrix = np.array([[ 1.18129821e+00,  1.01780132e-01, -1.01256520e+02],
#                         [-4.96726637e-02,  1.16948717e+00, -3.50429676e+01]])

# path = '/images/yolo+1/20240112_1450/*jpg'
# path = '/images/yolo+1/*/*jpg'
path = '/images/data/0731/table1/*jpg'

file_list = peekable(sorted(glob.iglob(path)))


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
bg_first_v = cv2.imread('/images/data/0731/table1/test_table1_V.jpg')
bg_first_v_gray = cv2.cvtColor(bg_first_v, cv2.COLOR_BGR2GRAY)
_, table_mask = cv2.threshold(bg_first_v_gray, 80, 255, cv2.THRESH_BINARY)


model = YOLO("yolov8x-seg.pt")
flag_hand = 0
color_g = {}


shutil.rmtree('../results/yolo_table1_thumbnails/')
shutil.rmtree('../results/yolo_table1_detail/')
shutil.rmtree('../results/yolo_table1_mask/')
os.remove('./log/yolo_table1.csv')


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

            # YOLO
            pred = model.predict(img_v_color, classes=[0, 25, 26, 28, 39, 41, 64, 65, 67, 73, 74, 76])
            frame = pred[0].plot()
            bboxes = pred[0].boxes.xyxy.cpu().numpy()
            classes = pred[0].boxes.cls.cpu().numpy()
            seg_masks = pred[0].masks
            
            
            # 人の手があるとき
            if 0 in classes: flag_hand = 1
            
            polygon = []
            for x1, y1, x2, y2 in bboxes:
                polygon.append(np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]))
            
            
            hue_mask = np.ones((480, 640), np.uint8)
            for (cx, cy), [hue, (x,y,w,h)] in list(color_g.items()):
                hue_img = cv2.cvtColor(img_v, cv2.COLOR_BGR2HSV)[:,:,0]
                
                # Hueの値が同じなら
                if hue_img[cy, cx] > hue - 10 and hue_img[cy, cx] < hue + 10:
                    polygon.append(np.array([[x, y], [x, y+h], [x+w, y+h], [x+w, y]]))
                    classes = np.append(classes, -1)
                    bboxes = np.vstack([bboxes, np.array([x, y, x+w, y+h])])
                    cv2.rectangle(hue_mask, (x, y), (x + w, y + h), (0, 0, 0), -1) #塗りつぶし
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) #描画
                else:
                    del color_g[(cx, cy)]
            
            
            if seg_masks is not None:
                # セグメンテーションのマスクを生成
                seg_mask = np.zeros((480, 640), np.uint8)
                for m in seg_masks.data.cpu().numpy():
                    m = m.astype(np.uint8)*255
                    seg_mask = cv2.bitwise_or(seg_mask, m)
                
                
                diff = cv2.absdiff(img_v, bg_first_v)
                diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                #YOLOで検知したものを塗りつぶし
                seg_mask = cv2.bitwise_not(seg_mask)
                seg_mask = cv2.dilate(seg_mask, kernel, 5)

                _, thres = cv2.threshold(diff,10,255,cv2.THRESH_BINARY)
                thres = cv2.bitwise_and(thres, seg_mask)
                thres = cv2.bitwise_and(thres, hue_mask)
                thres = cv2.bitwise_and(thres, table_mask)
                thres = cv2.erode(thres, kernel, 5)
                
                contours2, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for cnt in contours2:
                    area = cv2.contourArea(cnt)
                    # 面積が一定以上の場合にのみ矩形を描画
                    if area > 300:
                        x, y, w, h = cv2.boundingRect(cnt)
                        
                        x, y = max(0, x-10), max(0, y-10)
                        w = w + 20 if x + w + 20 < 640 else w + 10
                        h = h + 20 if y + h + 20 < 640 else h + 10
                        
                        polygon.append(np.array([[x, y], [x, y+h], [x+w, y+h], [x+w, y]]))
                        classes = np.append(classes, -1)
                        bboxes = np.vstack([bboxes, np.array([x, y, x+w, y+h])])
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) #描画
            
            
            if flag_hand != 1:
                with open("./log/yolo_table1.csv", "a") as f:
                    for cls, bbox in zip(classes, bboxes):
                        time = datetime.datetime.now()
                        data = [[time, "table1", int(cls), list(bbox)]]
                        writer = csv.writer(f)
                        writer.writerows(data)
                        
                        path = '../results/yolo_table1_thumbnails/{}'.format(int(cls))
                        if not os.path.exists(path): os.makedirs(path)
                        path = '../results/yolo_table1_detail/{}'.format(int(cls))
                        if not os.path.exists(path): os.makedirs(path)
                        path = '../results/yolo_table1_mask/{}'.format(int(cls))
                        if not os.path.exists(path): os.makedirs(path)
                        
                        xmin, ymin, xmax, ymax = map(int, bbox[:4])
                        crop = img_v_color[ymin:ymax, xmin:xmax]
                        cv2.imwrite('../results/yolo_table1_thumbnails/{}/{}.jpg'.format(int(cls),time), crop)
                        
                        overview = img_v_color.copy()
                        cv2.rectangle(overview, (xmin,ymin), (xmax,ymax), (0, 0, 255), thickness=5)
                        cv2.imwrite('../results/yolo_table1_detail/{}/detail_{}.jpg'.format(int(cls),time), overview)
                        
                        mask = cv2.bitwise_and(erode_v, erode_t)
                        mask_inv = cv2.bitwise_not(mask)[ymin:ymax, xmin:xmax]
                        cv2.imwrite('../results/yolo_table1_mask/{}/{}.jpg'.format(int(cls),time), mask_inv)

            # if (feature_compare(b_img_v, img_v)<12):
            if (0 not in classes and feature_compare(b_img_v, img_v)<12): #手がない時に背景更新
                bg_v = img_v.copy()
                b_img_v = img_v.copy()
                # bg_t = img_t.copy()
                
                # 手が無くなった直後
                if flag_hand == 1:
                    flag_hand = 0
                    
                    bg_v_noshade = delete_shade(bg_v)
                    bg_diff = cv2.bitwise_xor(bg_v_noshade, bg_before).astype(np.uint8) #1つ前の背景-今の背景
                    bg_diff = cv2.bitwise_and(seg_mask, bg_diff) #YOLOで検知されたものを除外
                    bg_diff = cv2.bitwise_and(table_mask, bg_diff) #tableの範囲のみで画像処理
                    bg_diff = cv2.erode(bg_diff, kernel, 1)
                    bg_diff = cv2.dilate(bg_diff, kernel, 1)
                    bg_before = bg_v_noshade
                    
                    contours_bg, _ = cv2.findContours(bg_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for cnt in contours_bg:
                        area = cv2.contourArea(cnt)
                        if area > 300:
                            x, y, w, h = cv2.boundingRect(cnt)
                            
                            x, y = max(0, x-10), max(0, y-10)
                            w = w + 20 if x + w + 20 < 640 else w + 10
                            h = h + 20 if y + h + 20 < 640 else h + 10                          
                            
                            cv2.rectangle(bg_diff, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            
                            M = cv2.moments(cnt)
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            hue = cv2.cvtColor(img_v, cv2.COLOR_BGR2HSV)[:,:,0]
                            
                            # 重心の色(Hue), bboxを保存
                            color_g[(cx, cy)] = [hue[cy, cx], (x,y,w,h)]
                        
                    
            else: b_img_v = img_v.copy()
            
    except StopIteration:
        break

# end = time.time()
# print(end-start)
