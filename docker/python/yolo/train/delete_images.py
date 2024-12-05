from tqdm import tqdm
from ultralytics.utils.checks import check_requirements
import numpy as np
from pathlib import Path
import random
import json

check_requirements(('pycocotools>=2.0',))
from pycocotools.coco import COCO


def delete_category_images(dir, target_classes):
    """
    target_classesに含まれないidのみで構成された画像を全て削除する
    """
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

        # 全てのカテゴリ
        names = [x["name"] for x in coco.loadCats(coco.getCatIds())]
        for cat in names: # 1カテゴリずつ処理
            catIds = coco.getCatIds(catNms=[cat])
            imgIds = coco.getImgIds(catIds=catIds)

            # 画像を処理
            for im in tqdm(coco.loadImgs(imgIds), desc=f'Processing images in {split}'):
                img_id = im["id"]
                img_path = images_dir / Path(im["file_name"]).name

                if img_id not in target_img_ids:
                    # 不要な画像を削除
                    if img_path.exists():
                        img_path.rename(dir / 'unnecessary_images' / split / Path(im["file_name"]).name)
                        # img_path.unlink()
                        print(f"Deleted image: {img_path}")

                    # 対応するラベルファイルも削除
                    label_path = labels_dir / img_path.with_suffix('.txt').name
                    if label_path.exists():
                        label_path.rename(dir / 'unnecessary_labels' / split / img_path.with_suffix('.txt').name)
                        # label_path.unlink()
                        print(f"Deleted label: {label_path}")



def filter_and_delete_images(original_json, image_dir, label_dir, output_json, delete_label, exclude_labels, ratio=0.5):
    """
    特定カテゴリ（Person）の画像を削除。ただし、指定されたカテゴリ（exclude_labels）が含まれる画像は削除しない。
    
    Args:
        original_json (str): 元のCOCOアノテーションファイルのパス
        image_dir (str): 画像ディレクトリ
        output_json (str): 更新後のアノテーションファイルの保存先
        delete_label (str): 削除対象カテゴリの名前
        exclude_labels (list): 削除したくないカテゴリの名前リスト
        ratio (float): 削除割合（0.0～1.0）
    """
    coco = COCO(original_json)

    # カテゴリIDを取得
    delete_id = coco.getCatIds(catNms=[delete_label])[0]
    exclude_cat_ids = set(coco.getCatIds(catNms=exclude_labels))

    # 各カテゴリに関連する画像IDを取得
    delete_img_ids = set(coco.getImgIds(catIds=[delete_id]))
    exclude_img_ids = set()
    for cat_id in exclude_cat_ids:
        exclude_img_ids.update(coco.getImgIds(catIds=[cat_id]))
        
    # 削除候補
    delete_candidates = delete_img_ids - exclude_img_ids
    delete_target = set(random.sample(delete_candidates, int(len(delete_candidates) * ratio)))

    # 新しい画像とアノテーションを構築
    filtered_images = []
    filtered_annotations = []

    all_image_ids = coco.getImgIds()
    all_images = coco.loadImgs(all_image_ids)
    for img in tqdm(all_images, desc="Filtering images"):
        if img["id"] not in delete_target:  # 削除対象でない場合に保持
            filtered_images.append(img)
            ann_ids = coco.getAnnIds(imgIds=[img["id"]])
            filtered_annotations.extend(coco.loadAnns(ann_ids))
    
    # 新しいアノテーションデータを保存
    filtered_data = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco.dataset["categories"],
    }

    with open(output_json, "w") as f:
        json.dump(filtered_data, f, indent=4)

    # 削除対象の画像ファイルを削除
    image_dir = Path(image_dir)
    dataset_dir = Path('/home/srv-admin/webapp/docker/python/yolo/datasets/Objects365')
    for img_id in tqdm(delete_target, desc="Deleting images"):
        img_info = coco.loadImgs([img_id])[0]
        img_path = image_dir / Path(img_info["file_name"]).name
        label_path = label_dir / img_path.with_suffix('.txt').name
        
        if img_path.exists():
            img_path.rename(dataset_dir / 'removed_images' / 'train' / Path(img_info["file_name"]).name)
            print(f"Deleted: {img_path}")
            
        if label_path.exists():
            label_path.rename(dataset_dir / 'removed_labels' / 'train' / img_path.with_suffix('.txt').name)
            print(f"Deleted label: {label_path}")

# 実行部分
dataset_dir = Path('/home/srv-admin/webapp/docker/python/yolo/datasets/Objects365')
split = "train"
filter_and_delete_images(
    original_json = dataset_dir / f"updated_annotations_{split}.json",
    image_dir = dataset_dir / "images" / split,
    label_dir = dataset_dir / "labels" / split,
    output_json = dataset_dir / f"updated_annotations_{split}.json",
    delete_label = "Book",
    exclude_labels = ["Remote", "Scissors", "Folder", "earphone", "Mask", "Tissue", "Wallet/Purse",
                    "Tablet", "Key", "CD", "Stapler", "Eraser", "Lipstick"],  # 削除したくないカテゴリをリストで指定
    ratio = 0.6
    )

# if __name__ == '__main__':
#     dir = Path('/home/srv-admin/webapp/docker/python/yolo/datasets/Objects365')
#     target_classes = ["Person", "Glasses", "Bottle", "Cup", "Handbag/Satchel", "Book",
#                     "Umbrella", "Watch", "Pen/Pencil", "Cell Phone",
#                     "Laptop", "Clock", "Keyboard", "Mouse", "Head Phone", "Remote",
#                     "Scissors", "Folder", "earphone", "Mask", "Tissue", "Wallet/Purse",
#                     "Tablet", "Key", "CD", "Stapler", "Eraser", "Lipstick"]
    
#     delete_category_images(dir, target_classes)


