import datetime
import csv
import cv2
import sys
import os


def calc_iou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    
    xmin_intersection = max(xmin1, xmin2)
    ymin_intersection = max(ymin1, ymin2)
    xmax_intersection = min(xmax1, xmax2)
    ymax_intersection = min(ymax1, ymax2)
    
    intersection_width = max(0, xmax_intersection - xmin_intersection)
    intersection_height = max(0, ymax_intersection - ymin_intersection)
    
    intersection_area = intersection_width * intersection_height
    
    # bbox1とbbox2の各領域の面積を計算
    bbox1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    bbox2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    
    # IoUを計算
    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)
    return iou

def calc_area(bbox):
    '''
    バウンディングボックスの面積を求める
    '''
    xmin, ymin, xmax, ymax = bbox
    area = (xmax-xmin)*(ymax-ymin)
    return area

def calc_hist(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img], [0], mask, [256], [0, 256])
    hist = cv2.normalize(hist, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return hist

def is_same_object(hist1, hist2, bbox1, bbox2):
    hist_comp = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    iou = calc_iou(bbox1, bbox2)
    return hist_comp + iou

def analysis(input_csv_path, output_csv_path, result_dir):
    """
    input: [time, day_of_week, place, label, list(bbox), is_touch]
    output: [id, label, arrival_time, stay_time, count, day_of_week, x1, y1, x2, y2, place, is_touch]
    """
    
    MIN_DATA_THRES = 50
    
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)
    label_dict = {} # label_count: first_time, last_time, count, bbox
    hist_dict = {} # label: histのリスト
    
    header = ["id", "label", "arrival_time", "stay_time", "count", "day_of_week", "x1", "y1", "x2", "y2", "place", "is_touch"]
    with open(output_csv_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    with open(input_csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            dow = row[1]
            place = row[2]
            label = row[3]
            df_time = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            bbox1 = [int(float(c)) for c in row[4][1:-1].split(',')]
            is_touch = row[5]
            
            # 今までに検出されなかったラベル
            if label not in hist_dict.keys():
                img = cv2.imread('../results/{}_thumbnails/{}/{}.jpg'.format(result_dir, label, df_time))
                mask = cv2.imread('../results/{}_mask/{}/{}.jpg'.format(result_dir, label, df_time))
                hist_dict[label] = [calc_hist(img, mask)]
                label_dict['{}_0'.format(label)] = [df_time, df_time, 1, bbox1]
            
            # 過去に検出されたラベルなら
            else:
                # 同一物体かを判定
                score = [] # 類似度スコア
                img = cv2.imread('../results/{}_thumbnails/{}/{}.jpg'.format(result_dir, label, df_time))
                mask = cv2.imread('../results/{}_mask/{}/{}.jpg'.format(result_dir, label, df_time))
                hist1 = calc_hist(img, mask)
                for i, hist2 in enumerate(hist_dict[label]):
                    bbox2 = label_dict['{}_{}'.format(label, i)][3]
                    score.append(is_same_object(hist1, hist2, bbox1, bbox2))
                
                # 過去に同一物体があるとき
                if max(score) > 1.1:
                    index = score.index(max(score)) # スコアが最も高いもののインデックス
                    
                    # ヒストグラム更新
                    hist_dict[label][index] = hist1
                    
                    label_index = '{}_{}'.format(label, index)
                    label_dict[label_index][2] += 1
                    bbox_avg = label_dict[label_index][3]
                    n = label_dict[label_index][2]
                    
                    #bboxの平均を更新
                    label_dict[label_index][3] = [round((n*y+x)/(n+1)) for x, y in zip(bbox1, bbox_avg)]
            
                    # 最終時刻と確認された時刻の差が一定以内であれば最終時刻を更新
                    if df_time - label_dict[label_index][1] < datetime.timedelta(seconds=10):
                        label_dict[label_index][1] = df_time

                    #物体が一定時間を超えて再び確認された場合
                    else:
                        with open(output_csv_path, 'a', newline='') as w:
                            writer = csv.writer(w)
                            # label_index: first_time, last_time, count, bbox
                            if (label_dict[label_index][2] > MIN_DATA_THRES):
                                x1, y1, x2, y2 = label_dict[label_index][3]
                                l = label_index.partition('_')
                                first_time = label_dict[label_index][0]
                                last_time = label_dict[label_index][1]
                                elapsed_time = last_time - first_time
                                elapsed_time = elapsed_time.total_seconds()
                                line = [label_index] + [l[0]] + [first_time] + [int(elapsed_time)] + \
                                            [label_dict[label_index][2]] + [dow] + [x1, y1, x2, y2] + [place] + [is_touch]
                                writer.writerow(line)
                        
                        label_dict[label_index] = [df_time, df_time, 1, bbox1]
                        

                # 過去に同一物体がないとき
                else:
                    label_dict['{}_{}'.format(label, len(hist_dict[label]))] = [df_time, df_time, 1, bbox1]
                    hist_dict[label].append(hist1)
                
        with open(output_csv_path, 'a', newline="") as w:
            writer = csv.writer(w)
            for label_index in label_dict.keys():
                if (label_dict[label_index][2] > MIN_DATA_THRES):
                    x1, y1, x2, y2 = label_dict[label_index][3]
                    l = label_index.partition('_')
                    first_time = label_dict[label_index][0]
                    last_time = label_dict[label_index][1]
                    elapsed_time = last_time - first_time
                    elapsed_time = elapsed_time.total_seconds()
                    line = [label_index] + [l[0]] + [first_time] + [int(elapsed_time)] + \
                                [label_dict[label_index][2]] + [dow] + [x1, y1, x2, y2] + [place] + [is_touch]
                    writer.writerow(line)

if __name__ == '__main__':
    # python analysis.py input.csv output.csv result_dir
    input_csv_path = './log/' + sys.argv[1]
    output_csv_path = './analysis/' + sys.argv[2]
    analysis(input_csv_path, output_csv_path, result_dir=sys.argv[3])
