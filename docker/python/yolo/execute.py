from analysis import analysis
from db_controll import delete_all_data, insert_csv_data
import argparse
import mysql.connector

# コネクションの作成
conn = mysql.connector.connect(
    host='mysql',
    port='3306',
    user='root',
    password='root',
    database='thermal'
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('-dir', '--result_dir')
    
    args = parser.parse_args()
    
    analysis('./log/' + args.input, './analysis/' + args.output, args.result_dir)
    delete_all_data()
    insert_csv_data('./analysis/' + args.output)