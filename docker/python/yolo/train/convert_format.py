from pathlib import Path
from pycocotools.coco import COCO
import numpy as np
from ultralytics.utils.ops import xyxy2xywhn


def convert_coco_to_yolo_segmentation(coco_json, output_dir):
    """
    COCO形式のセグメンテーションデータをYOLO形式に変換する。
    
    Parameters:
        coco_json (str or Path): フィルタリング済みのCOCO形式のアノテーションファイル
        output_dir (str or Path): YOLO形式のラベルを保存するディレクトリ
    """
    coco = COCO(coco_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        for image in coco.loadImgs(coco.getImgIds()):
            label_file = output_dir / f"{split}" / f"{Path(image['file_name'])}.txt"
            with open(label_file, "w") as f:
                ann_ids = coco.getAnnIds(imgIds=[image['id']])
                annotations = coco.loadAnns(ann_ids)

                for ann in annotations:
                    # セグメンテーションポイントを取得
                    segmentation = ann.get('segmentation', [])
                    if isinstance(segmentation, list):  # Polygons形式を想定
                        segmentation_points = ' '.join(map(str, segmentation[0]))  # 1つのリストにする

                    width, height = image["width"], image["height"]
                    x, y, w, h = ann['bbox']
                    xyxy = np.array([x, y, x + w, y + h])[None]  # pixels(1,4)
                    x, y, w, h = xyxy2xywhn(xyxy, w=width, h=height, clip=True)[0]  # normalized and clipped
                    
                    class_id = ann['category_id'] - 1  # YOLOではクラスIDは0ベース

                    # YOLO形式で書き込む
                    f.write(f"{class_id} {x:.5f} {y:.5f} {w:.5f} {h:.5f} {segmentation_points}\n")

        print(f"Converted annotations saved to: {output_dir}")


dir = Path('/home/srv-admin/webapp/docker/python/yolo')
coco_json = Path(dir/"filtered_seg.json")
output_dir = Path(dir/"datasets/Objects365/converted")
convert_coco_to_yolo_segmentation(coco_json, output_dir)
