import datetime
import csv

csv_path = 'result.csv'
# df_dict = {}
label_dict = {}

with open(csv_path) as f:
    reader = csv.reader(f)
    for row in reader:
        label = row[2]
        
        if label not in label_dict.keys():
            df_time = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
            label_dict[label] = [df_time, df_time, 1, [float(c) for c in row[3][1:-1].split(',')]]
            
        else:
            label_dict[label][2] += 1
            bbox = [float(c) for c in row[3][1:-1].split(',')]
            bbox_avg = label_dict[label][3]
            n = label_dict[label][2]
            
            #bboxの平均を更新
            label_dict[label][3] = [round((n*y+x)/(n+1)) for x, y in zip(bbox, bbox_avg)]
            
            #最後に物体が確認された時間
            last_time = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
    
            if last_time - label_dict[label][1] < datetime.timedelta(seconds=30):
                label_dict[label][1] = last_time

            #物体が一定時間を超えて再び確認された場合
            else:
                with open('analysis.csv', 'a', newline='') as w:
                    writer = csv.writer(w)
                    x1, y1, x2, y2 = label_dict[label][3]
                    line = [label] + label_dict[label][0:3] + [x1, y1, x2, y2]
                    writer.writerow(line)
                
                label_dict[label] = [last_time, last_time, 1, [float(c) for c in row[3][1:-1].split(',')]]

    with open('analysis.csv', 'a', newline="") as w:
        writer = csv.writer(w)
        for label in label_dict.keys():
            x1, y1, x2, y2 = label_dict[label][3]
            line = [label] + label_dict[label][0:3] + [x1, y1, x2, y2]
            writer.writerow(line)
    

