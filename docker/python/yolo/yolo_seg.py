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

# start = time.time()
# affine_matrix = np.array([[ 1.15775321e+00, 2.06036561e-02, -8.65530736e+01],
#                         [-3.59868529e-02, 1.16843440e+00, -4.39524932e+01]])

affine_matrix = np.array([[1.15919938e+00, 7.27146534e-02, -5.70173323e+01],
                        [1.46108543e-04, 1.16974505e+00, -5.52789456e+01]])


# path = '/images/yolo+1/20240112_1450/*jpg'
# path = '/images/yolo+1/*/*jpg'
path = '/images/data/0701/table/*jpg'

file_list = peekable(sorted(glob.iglob(path)))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('test_table_0701.mp4',fourcc, 30.3, (640, 480), isColor=True)

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

# 1つ前の背景
bg_before = delete_shade(bg_v)
kernel = np.ones((5,5), np.uint8)

# 初期背景（何も置かれていない）
# bg_first_v = bg_v.copy()
# bg_first_v_gray = cv2.cvtColor(bg_first_v, cv2.COLOR_BGR2GRAY)
bg_first_v = cv2.imread('/images/data/0701/test_table_V.jpg')
bg_first_v_gray = cv2.cvtColor(bg_first_v, cv2.COLOR_BGR2GRAY)
_, table_mask = cv2.threshold(bg_first_v_gray, 80, 255, cv2.THRESH_BINARY)

bg_first_v = bg_t.copy()

model = YOLO("yolov8x-seg.pt")
flag_hand = 0
color_g = {}


# shutil.rmtree('../results/table_0527_thumbnails/')
# shutil.rmtree('../results/table_0527_detail/')
# shutil.rmtree('../results/table_0527_mask/')
# os.remove('./log/test_table_0527.csv')


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

                _, thres = cv2.threshold(diff,20,255,cv2.THRESH_BINARY)
                thres = cv2.bitwise_and(thres, seg_mask)
                thres = cv2.bitwise_and(thres, hue_mask)
                thres = cv2.bitwise_and(thres, table_mask)
                thres = cv2.erode(thres, kernel, 5)
                
                contours2, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for cnt in contours2:
                    area = cv2.contourArea(cnt)
                    # 面積が一定以上の場合にのみ矩形を描画
                    if area > 500:
                        x, y, w, h = cv2.boundingRect(cnt)
                        
                        x, y = max(0, x-10), max(0, y-10)
                        w = w + 20 if x + w + 20 < 640 else w + 10
                        h = h + 20 if y + h + 20 < 640 else h + 10
                        
                        polygon.append(np.array([[x, y], [x, y+h], [x+w, y+h], [x+w, y]]))
                        classes = np.append(classes, -1)
                        bboxes = np.vstack([bboxes, np.array([x, y, x+w, y+h])])
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) #描画
            
            mask = touch_region
            mask_inv = cv2.bitwise_not(mask)
            back = cv2.bitwise_and(frame, frame, mask_inv)
            cut = cv2.bitwise_and(mask, mask, mask)
            cut = cv2.cvtColor(cut, cv2.COLOR_GRAY2BGR)
            paste = cv2.add(back, cut)
            

            # with open("./log/test_table3_0529.csv", "a") as f:
            #     for poly, cls, bbox in zip(polygon, classes, bboxes):
            #         for pt in points:
            #             if cv2.pointPolygonTest(poly, pt, False) >= 0 and cls != 0:
            #                 time = datetime.datetime.now()
            #                 data = [[time, "table", int(cls), list(bbox)]]
            #                 writer = csv.writer(f)
            #                 writer.writerows(data)
                            
            #                 path = '../results/table3_0529_thumbnails/{}'.format(int(cls))
            #                 if not os.path.exists(path): os.makedirs(path)
            #                 path = '../results/table3_0529_detail/{}'.format(int(cls))
            #                 if not os.path.exists(path): os.makedirs(path)
            #                 path = '../results/table3_0529_mask/{}'.format(int(cls))
            #                 if not os.path.exists(path): os.makedirs(path)
                            
            #                 xmin, ymin, xmax, ymax = map(int, bbox[:4])
            #                 crop = img_v_color[ymin:ymax, xmin:xmax]
            #                 cv2.imwrite('../results/table3_0529_thumbnails/{}/{}.jpg'.format(int(cls),time), crop)
                            
            #                 overview = img_v_color.copy()
            #                 cv2.rectangle(overview, (xmin,ymin), (xmax,ymax), (0, 0, 255), thickness=5)
            #                 cv2.imwrite('../results/table3_0529_detail/{}/detail_{}.jpg'.format(int(cls),time), overview)
                            
            #                 mask = cv2.bitwise_and(erode_v, erode_t)
            #                 mask_inv = cv2.bitwise_not(mask)[ymin:ymax, xmin:xmax]
            #                 cv2.imwrite('../results/table3_0529_mask/{}/{}.jpg'.format(int(cls),time), mask_inv)
            #                 break

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
                        if area > 500:
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
            
            video.write(paste)

    except StopIteration:
        break

video.release()
# end = time.time()
# print(end-start)
