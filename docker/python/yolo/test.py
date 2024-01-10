import mysql.connector
from datetime import datetime

  # コネクションの作成
conn = mysql.connector.connect(
    host='mysql',
    port='3306',
    user='root',
    password='root',
    database='thermal'
)

# コネクションが切れた時に再接続してくれるよう設定
conn.ping(reconnect=True)

# 接続できているかどうか確認
# print(conn.is_connected())

cur = conn.cursor()
with open('analysis.csv') as f:
    id = 1
    for line in f:
        # 読み込んだ行の項目を順にカンマ区切りで対応する変数へ文字列としてmapする。
        label, first_time, last_time, count, bbox_x1, bbox_y1, bbox_x2, bbox_y2 = map(str, line.split(','))
        
        cur.execute("""INSERT INTO csv (id, label, first_time, last_time, count, bbox_x1, bbox_y1, bbox_x2, bbox_y2) 
                    values(%s,%s,%s,%s,%s,%s,%s,%s,%s);""", (id, label, first_time, last_time, count, bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        
        id += 1

cur.close()
conn.commit()
conn.close()
