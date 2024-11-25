import json
import random
from pathlib import Path
from pycocotools.coco import COCO

def create_subset(original_json, output_json, target_per_category, target_classes):
    """
    各カテゴリごとに指定した枚数の画像を使用したサブセットを作成します。

    Parameters:
        original_json (str or Path): 元のアノテーションファイル（COCO形式）
        output_json (str or Path): サブセットのアノテーションファイルの出力先
        target_per_category (int): 各カテゴリで使用する画像数
        target_classes (list): 対象カテゴリの名前リスト
    """

    coco = COCO(original_json)
    
    # 対象カテゴリのIDを取得
    target_cat_ids = coco.getCatIds(catNms=target_classes)
    print(f"Target Category IDs: {target_cat_ids}")

    # サブセット用の画像IDとアノテーションを収集
    subset_img_ids = set()
    for cat_id in target_cat_ids:
        img_ids = coco.getImgIds(catIds=[cat_id])
        sampled_img_ids = random.sample(img_ids, min(target_per_category, len(img_ids)))
        subset_img_ids.update(sampled_img_ids)

    # フィルタリングされた画像とアノテーションを収集
    filtered_images = [img for img in coco.loadImgs(list(subset_img_ids))]
    filtered_annotations = [
        ann for ann in coco.loadAnns(coco.getAnnIds(imgIds=list(subset_img_ids)))
    ]
    filtered_categories = [cat for cat in coco.loadCats(target_cat_ids)]

    # 新しいアノテーションデータを構築
    subset_data = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": filtered_categories,
    }

    # JSONとして保存
    with open(output_json, "w") as f:
        json.dump(subset_data, f, indent=4)

    print(f"Subset annotations saved to {output_json}")
    print(f"Number of images: {len(filtered_images)}")
    print(f"Number of annotations: {len(filtered_annotations)}")


dir = Path('/home/srv-admin/webapp/docker/python/yolo/datasets/Objects365')
original_json = dir / "/filtered_annotations_train.json"
output_json = dir / "/subset_annotations_train.json"
target_per_category = 100
target_classes = ["Person", "Glasses", "Bottle", "Cup", "Handbag/Satchel", "Book",
                        "Umbrella", "Watch", "Pen/Pencil", "Cell Phone",
                        "Laptop", "Clock", "Keyboard", "Mouse", "Head Phone", "Remote",
                        "Scissors", "Folder", "earphone", "Mask", "Tissue", "Wallet/Purse",
                        "Tablet", "Key", "CD", "Stapler", "Eraser", "Lipstick"]

create_subset(original_json, output_json, target_per_category, target_classes)
