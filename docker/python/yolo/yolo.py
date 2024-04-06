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
    

affine_matrix = np.array([[ 1.15775321e+00, 2.06036561e-02, -8.65530736e+01],
                        [-3.59868529e-02, 1.16843440e+00, -4.39524932e+01]])

path = '/images/items*/*/*.jpg'

# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# video = cv2.VideoWriter('log.mp4',fourcc, 30.3, (640, 480))

def yolo(path=path):
    # shutil.rmtree('../results/thumbnails/')
    # shutil.rmtree('../results/detail/')
    # shutil.rmtree('../results/mask/')
    # os.remove('result.csv')
    
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
                
                # frame = pred[0].plot()
                # mask_inv = cv2.bitwise_not(touch_region)
                # back = cv2.bitwise_and(frame, frame, mask_inv)
                # cut = cv2.bitwise_and(touch_region, touch_region, touch_region)
                # cut = cv2.cvtColor(cut, cv2.COLOR_GRAY2BGR)
                # paste = cv2.add(back, cut)
                
                polygon = []
                for x1, y1, x2, y2 in bboxes:
                    polygon.append(np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]))

                with open("yolo_only.csv", "a") as f:
                    for poly, cls, bbox in zip(polygon, classes, bboxes):
                        #物体検知履歴
                        time = datetime.datetime.now()
                        data = [[time, "table", int(cls), list(bbox)]]
                        writer = csv.writer(f)
                        writer.writerows(data)
                        
                        path = '../results/yolo_thumbnails/{}'.format(int(cls))
                        if not os.path.exists(path): os.makedirs(path)
                        path = '../results/yolo_detail/{}'.format(int(cls))
                        if not os.path.exists(path): os.makedirs(path)
                        path = '../results/yolo_mask/{}'.format(int(cls))
                        if not os.path.exists(path): os.makedirs(path)
                        
                        xmin, ymin, xmax, ymax = map(int, bbox[:4])
                        crop = img_v_color[ymin:ymax, xmin:xmax]
                        cv2.imwrite('../results/yolo_thumbnails/{}/{}.png'.format(int(cls),time), crop)
                        
                        overview = img_v_color.copy()
                        cv2.rectangle(overview, (xmin,ymin), (xmax,ymax), (0, 0, 255), thickness=5)
                        cv2.imwrite('../results/yolo_detail/{}/detail_{}.png'.format(int(cls),time), overview)
                        
                        mask = cv2.bitwise_and(erode_v, erode_t)
                        mask_inv = cv2.bitwise_not(mask)[ymin:ymax, xmin:xmax]
                        cv2.imwrite('../results/yolo_mask/{}/{}.png'.format(int(cls),time), mask_inv)
                        
                        #接触履歴
                        # for pt in points:
                        #     if cv2.pointPolygonTest(poly, pt, False) >= 0:
                        #         time = datetime.datetime.now()
                        #         data = [[time, "table", int(cls), list(bbox)]]
                        #         writer = csv.writer(f)
                        #         writer.writerows(data)
                                
                        #         path = '../results/thumbnails/{}'.format(int(cls))
                        #         if not os.path.exists(path): os.makedirs(path)
                        #         path = '../results/detail/{}'.format(int(cls))
                        #         if not os.path.exists(path): os.makedirs(path)
                        #         path = '../results/mask/{}'.format(int(cls))
                        #         if not os.path.exists(path): os.makedirs(path)
                                
                        #         xmin, ymin, xmax, ymax = map(int, bbox[:4])
                        #         crop = img_v_color[ymin:ymax, xmin:xmax]
                        #         cv2.imwrite('../results/thumbnails/{}/{}.png'.format(int(cls),time), crop)
                                
                        #         overview = img_v_color.copy()
                        #         cv2.rectangle(overview, (xmin,ymin), (xmax,ymax), (0, 0, 255), thickness=5)
                        #         cv2.imwrite('../results/detail/{}/detail_{}.png'.format(int(cls),time), overview)
                                
                        #         mask = cv2.bitwise_and(erode_v, erode_t)
                        #         mask_inv = cv2.bitwise_not(mask)[ymin:ymax, xmin:xmax]
                        #         cv2.imwrite('../results/mask/{}/{}.png'.format(int(cls),time), mask_inv)
                        #         break

                if (feature_compare(b_img_v, img_v)<12):
                    bg_v = img_v.copy()
                    # print('更新')
                else: b_img_v = img_v.copy()
                
        except StopIteration:
            break
        
if __name__ == '__main__':
    yolo()