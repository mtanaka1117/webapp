from ultralytics.data.annotator import auto_annotate
auto_annotate(data="./datasets/Objects365", det_model="yolov8x.pt", sam_model="sam_b.pt")

# coco = COCO(dir / f'filtered_annotations_train.json')
# print(f"Number of images: {len(coco.loadImgs())}")
# print(f"Number of annotations: {len(coco.loadAnns(coco.getAnnIds()))}")