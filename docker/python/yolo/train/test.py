from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO

dataset_dir = Path('/home/srv-admin/webapp/docker/python/yolo/datasets/Objects365')

# for split in ['train', 'val']:
#     print(f"Processing {split} ...")
    # unne_images = dataset_dir / 'unnecessary_images' / split
    # images = dataset_dir / 'images' / split
    # unne_labels = dataset_dir / 'unnecessary_labels' / split
    # rename = dataset_dir / 'unnecessary_labels' / split
    # removed_images = dataset_dir / 'removed_images' / split
    # removed_labels = dataset_dir / 'removed_labels' / split
    # rename_images = dataset_dir / 'images' / split
    # rename_labels = dataset_dir / 'labels' / split

    # for f in tqdm(unne_images.rglob('*.jpg'), desc=f'Moving {split} images'):
    #     f.rename(images / f.name)  
    
    # for f in tqdm(unne_labels.rglob('*.jpg'), desc=f'Moving {split} images'):
    #     f.rename(rename / f.with_suffix('.txt').name)

    # for f in tqdm(removed_images.rglob('*.jpg'), desc=f'Moving {split} images'):
    #     f.rename(rename_images / f.name)
    
    # for f in tqdm(removed_labels.rglob('*.txt'), desc=f'Moving {split} labels'):
    #     f.rename(rename_labels / f.name)

target_categories = ["Person", "Glasses", "Bottle", "Cup", "Handbag/Satchel", "Book",
                    "Umbrella", "Watch", "Pen/Pencil", "Cell Phone",
                    "Laptop", "Clock", "Keyboard", "Mouse", "Head Phone", "Remote",
                    "Scissors", "Folder", "earphone", "Mask", "Tissue", "Wallet/Purse",
                    "Tablet", "Key", "CD", "Stapler", "Eraser", "Lipstick"]

# coco = COCO(dataset_dir / "filtered_annotations_train.json")
coco = COCO(dataset_dir / "updated_annotations_train.json")
target_cat_ids = coco.getCatIds(catNms=target_categories)

for cat_id in target_cat_ids:
    print(f"Processing category: {cat_id}")
    target_img_ids = coco.getImgIds(catIds=[cat_id])
    print(len(target_img_ids))
