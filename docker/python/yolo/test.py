from yolo import yolo
import db_controll
from analysis import analysis


if __name__ == '__main__':
    yolo(path = '/images/items*/*/*.jpg')
    analysis(csv_path = 'result.csv')
    db_controll.delete_all_data()
    db_controll.insert_csv_data()