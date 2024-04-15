import mysql.connector
import sys

# コネクションの作成
conn = mysql.connector.connect(
    host='mysql',
    port='3306',
    user='root',
    password='root',
    database='thermal'
)

def insert_csv_data(input_csv, conn=conn):
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

  # with open('analysis_no_hist.csv') as f:
  with open(input_csv) as f:
      for line in f:
          label, first_time, last_time, count, bbox_x1, bbox_y1, bbox_x2, bbox_y2 = map(str, line.split(','))
          cur.execute("""INSERT INTO csv (id, label, first_time, last_time, count, bbox_x1, bbox_y1, bbox_x2, bbox_y2) 
                      values(%s,%s,%s,%s,%s,%s,%s,%s,%s);""", (last_id, label, first_time, last_time, count, bbox_x1, bbox_y1, bbox_x2, bbox_y2))
          # label, first_time, last_time, count, bbox_x1, bbox_y1, bbox_x2, bbox_y2, thumb_time, max_area = map(str, line.split(','))
          # cur.execute("""INSERT INTO csv (id, label, first_time, last_time, count, bbox_x1, bbox_y1, bbox_x2, bbox_y2, thumb_time, max_area) 
          #             values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);""", (last_id, label, first_time, last_time, count, bbox_x1, bbox_y1, bbox_x2, bbox_y2, thumb_time, max_area))
          last_id += 1

  cur.close()
  conn.commit()
  conn.close()


def delete_all_data(conn=conn):
  conn.ping(reconnect=True)
  cur = conn.cursor()
  query = "TRUNCATE TABLE csv"
  cur.execute(query)
  
  cur.close()
  conn.commit()
  conn.close()
  
  
def insert_image_data(time, path, conn=conn):
  conn.ping(reconnect=True)
  cur = conn.cursor()
  query = "SELECT id FROM csv ORDER BY id DESC LIMIT 1"
  cur.execute(query)
  result = cur.fetchone()
  last_id = result[0]+1 if result else 1
  
  cur.execute("""INSERT INTO image_path (id, time, path)
              values(%s, %s, %s);""", (last_id, time, path)) 
  
  cur.close()
  conn.commit()
  conn.close()

if __name__ == "__main__":
  delete_all_data()
  insert_csv_data(input_csv=sys.argv[1])