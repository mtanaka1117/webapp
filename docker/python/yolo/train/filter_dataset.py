import json
from pathlib import Path
from pycocotools.coco import COCO

# [1, 8, 9, 11, 14, 19, 40, 43, 55, 62, 74, 95, 107, 116, 126, 133, 170, 206, 208, 209, 226, 239, 244, 252, 285, 307, 358, 362]
def filter_annotations_chunked(original_json, output_json, target_categories, chunk_size=1):
    coco = COCO(original_json)
    target_cat_ids = coco.getCatIds(catNms=target_categories)
    
    for cid in target_cat_ids:
        img_ids = coco.getImgIds(catIds=[cid])
        print(f"Category ID {cid} has {len(img_ids)} images")
    
    # カテゴリIDを小分けに分割
    # chunks = [target_cat_ids[i:i + chunk_size] for i in range(0, len(target_cat_ids), chunk_size)]
    
    # filtered_images = []
    # filtered_annotations = []
    # filtered_categories = []

    # # 各チャンクごとに処理
    # for chunk in chunks:
    #     print(f"Processing chunk: {chunk}")
    #     target_img_ids = coco.getImgIds(catIds=chunk)
        
    #     # フィルタされたデータを追加
    #     filtered_images.extend([img for img in coco.loadImgs(target_img_ids)])
    #     filtered_annotations.extend([
    #         ann for ann in coco.loadAnns(coco.getAnnIds(imgIds=target_img_ids, catIds=chunk))
    #     ])
    #     filtered_categories.extend([cat for cat in coco.loadCats(chunk)])
    #     print(len(filtered_images))

    # # 重複を削除
    # filtered_images = {img['id']: img for img in filtered_images}.values()
    # filtered_annotations = {ann['id']: ann for ann in filtered_annotations}.values()
    # filtered_categories = {cat['id']: cat for cat in filtered_categories}.values()

    # # 結果をJSONとして保存
    # filtered_data = {
    #     "images": list(filtered_images),
    #     "annotations": list(filtered_annotations),
    #     "categories": list(filtered_categories)
    # }
    # with open(output_json, "w") as f:
    #     json.dump(filtered_data, f, indent=4)



# 実行
dir = Path('/home/srv-admin/webapp/docker/python/yolo/datasets/Objects365')
split = 'train'  # 'train' または 'val'
target_categories = ["Person", "Glasses", "Bottle", "Cup", "Handbag/Satchel", "Book",
                    "Umbrella", "Watch", "Pen/Pencil", "Cell Phone",
                    "Laptop", "Clock", "Keyboard", "Mouse", "Head Phone", "Remote",
                    "Scissors", "Folder", "earphone", "Mask", "Tissue", "Wallet/Purse",
                    "Tablet", "Key", "CD", "Stapler", "Eraser", "Lipstick"]

filter_annotations_chunked(
    original_json=dir / f'zhiyuan_objv2_{split}.json',
    output_json=dir / f'filtered_annotations_{split}.json',
    # original_json='o365_instance_seg_val.json',
    # output_json='filtered_seg.json',
    target_categories=target_categories,
    chunk_size=1
)


