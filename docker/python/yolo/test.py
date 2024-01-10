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

# 最後のIDが何番目か取得する
query = "SELECT id FROM csv ORDER BY id DESC LIMIT 1"
cur.execute(query)

result = cur.fetchone()
last_id = result[0]+1 if result else 1

with open('analysis.csv') as f:
    for line in f:
        # 読み込んだ行の項目を順にカンマ区切りで対応する変数へ文字列としてmapする。
        label, first_time, last_time, count, bbox_x1, bbox_y1, bbox_x2, bbox_y2 = map(str, line.split(','))
        
        cur.execute("""INSERT INTO csv (id, label, first_time, last_time, count, bbox_x1, bbox_y1, bbox_x2, bbox_y2) 
                    values(%s,%s,%s,%s,%s,%s,%s,%s,%s);""", (last_id, label, first_time, last_time, count, bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        last_id += 1

cur.close()
conn.commit()
conn.close()
