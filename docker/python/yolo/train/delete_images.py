from tqdm import tqdm
from ultralytics.utils.checks import check_requirements
import numpy as np
from pathlib import Path

check_requirements(('pycocotools>=2.0',))
from pycocotools.coco import COCO

# Objects365 dataset のうち、不要なデータを削除する
dir = Path('/home/srv-admin/webapp/docker/python/yolo/datasets/Objects365')
target_classes = ["Person", "Glasses", "Bottle", "Cup", "Handbag/Satchel", "Book",
                "Umbrella", "Watch", "Pen/Pencil", "Cell Phone",
                "Laptop", "Clock", "Keyboard", "Mouse", "Head Phone", "Remote",
                "Scissors", "Folder", "earphone", "Mask", "Tissue", "Wallet/Purse",
                "Tablet", "Key", "CD", "Stapler", "Eraser", "Lipstick"]


for split in ['train', 'val']:
    print(f"Processing {split} ...")
    images_dir = dir / 'images' / split
    labels_dir = dir / 'labels' / split

    # COCO形式のアノテーションを読み込む
    coco = COCO(dir / f'zhiyuan_objv2_{split}.json')
    target_cat_ids = coco.getCatIds(catNms=target_classes)

    # 対象クラスに関連する画像IDを取得
    target_img_ids = set()
    for cat_id in target_cat_ids:
        target_img_ids.update(coco.getImgIds(catIds=[cat_id]))

    
    names = [x["name"] for x in coco.loadCats(coco.getCatIds())]
    for cid, cat in enumerate(names):
        catIds = coco.getCatIds(catNms=[cat])
        imgIds = coco.getImgIds(catIds=catIds)

        # 画像を処理
        for im in tqdm(coco.loadImgs(imgIds), desc=f'Processing images in {split}'):
            img_id = im["id"]
            img_path = images_dir / Path(im["file_name"]).name

            if img_id not in target_img_ids:
                # 不要な画像を削除
                if img_path.exists():
                    img_path.unlink()  # 画像ファイルを削除
                    print(f"Deleted image: {img_path}")

                # 対応するラベルファイルも削除
                label_path = labels_dir / img_path.with_suffix('.txt').name
                if label_path.exists():
                    label_path.unlink()  # ラベルファイルを削除
                    print(f"Deleted label: {label_path}")
