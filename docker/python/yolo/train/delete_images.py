from tqdm import tqdm
from ultralytics.utils.checks import check_requirements
import numpy as np
from pathlib import Path
import random

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


def remove_excess_images(coco_json, image_dir, label_dir, target_category, remove_ratio=0.5):
    """
    特定カテゴリに関連する画像をランダムに削除
    
    Parameters:
        coco_json (str or Path): COCO形式のアノテーションファイル
        image_dir (str or Path): 対応する画像ディレクトリ
        label_dir (str or Path): 対応するラベルディレクトリ
        target_category (str): 削除対象カテゴリ名
        remove_ratio (float): 削除する割合 (0.0〜1.0)
    """
    coco = COCO(coco_json)
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    # 削除対象カテゴリのIDを取得
    target_cat_ids = coco.getCatIds(catNms=target_category)
    if not target_cat_ids:
        print(f"Category '{target_category}' not found in the dataset.")
        return

    # 対象カテゴリに関連する画像IDを取得
    target_img_ids = coco.getImgIds(catIds=target_cat_ids)
    print(f"Found {len(target_img_ids)} images for category '{target_category}'.")

    # 削除する画像をランダムに選択
    num_to_remove = int(len(target_img_ids) * remove_ratio)
    img_ids_to_remove = set(random.sample(target_img_ids, num_to_remove))
    print(f"Removing {len(img_ids_to_remove)} images.")

    # 画像とラベルを削除
    removed_count = 0
    for img_id in tqdm(img_ids_to_remove):
        im = coco.loadImgs([img_id])[0]
        img_path = image_dir / Path(im["file_name"]).name
        label_path = label_dir / img_path.with_suffix('.txt').name

        if img_path.exists():
            # img_path.unlink()
            img_path.rename(dir / 'removed_images' / Path(im["file_name"]).name)
            removed_count += 1

        if label_path.exists():
            # label_path.unlink()
            label_path.rename(dir / 'removed_labels' / Path(im["file_name"]).name)

    print(f"Removed {removed_count} images and corresponding labels.")


if __name__ == '__main__':
    dir = Path('/home/srv-admin/webapp/docker/python/yolo/datasets/Objects365')
    target_classes = ["Person", "Glasses", "Bottle", "Cup", "Handbag/Satchel", "Book",
                    "Umbrella", "Watch", "Pen/Pencil", "Cell Phone",
                    "Laptop", "Clock", "Keyboard", "Mouse", "Head Phone", "Remote",
                    "Scissors", "Folder", "earphone", "Mask", "Tissue", "Wallet/Purse",
                    "Tablet", "Key", "CD", "Stapler", "Eraser", "Lipstick"]
    
    delete_category_images(dir, target_classes)



# if __name__ == '__main__':
#     dir = Path('/home/srv-admin/webapp/docker/python/yolo/datasets/Objects365')
    
#     coco_json = dir / "/filtered_annotations_train.json"
#     image_dir = dir / "/images/train"
#     label_dir = dir / "/labels/train"
#     target_category = ["Person"]
#     remove_excess_images(coco_json, image_dir, label_dir, target_category, remove_ratio=0.5)


