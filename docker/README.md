# Quick Start
```
cd docker
docker-compose up -d
```

# Code

## Detection
python/yolo
結果はlogフォルダに保存

```yolo_dev.py```: detectionを使用、
```yolo_only.py```: detectionを使用、サーマルカメラの情報なし
```yolo_seg_dev.py```: 開発中ver.
```yolo_seg.py```: segmentationを使用、差分画像処理あり、yolo_seg_dev.pyの完成版
```no_touch_detection.py```: segmentationを使用、差分画像処理あり、サーマルカメラの情報なし
```yolo_detect_obj365.py```: objects365 datasetでtrainしたYOLOを使用

```
python3 [.py]
```

## Analysis
結果はanalysisフォルダに保存

```analysis_dev.py```: 開発中ver.
```analysis.py```: 
```analysis_no_hist.py```: ヒストグラムを使用しない場合
```analysis_no_touch.py```: サーマルカメラの情報なし



## Database
```db_controll.py```: データベースからデータを全削除、登録する


## Execute
```execute.py```: analysis, databaseを一括で実行
```execute_no_touch.py```: 

## train
python/yolo/train
```download_dataset.py```: Objects365データセットをダウンロード
```delete_images.py```: 不要なラベルのみを持つ画像を unnecessary_images に移動
```convert_label.py```: ラベルをYOLOの形式に変換
```train_yolo.py```: 


# メモ
過去のresultsはNASとSSDに保存
