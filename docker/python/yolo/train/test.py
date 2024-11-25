from tqdm import tqdm
from pathlib import Path

dataset_dir = Path('/home/srv-admin/webapp/docker/python/yolo/datasets/Objects365')

for split in ['train', 'val']:
    print(f"Processing {split} ...")
    # unne_images = dataset_dir / 'unnecessary_images' / split
    # images = dataset_dir / 'images' / split
    unne_labels = dataset_dir / 'unnecessary_labels' / split
    rename = dataset_dir / 'unnecessary_labels' / split

    # for f in tqdm(unne_images.rglob('*.jpg'), desc=f'Moving {split} images'):
    #     f.rename(images / f.name)  
    
    for f in tqdm(unne_labels.rglob('*.jpg'), desc=f'Moving {split} images'):
        f.rename(rename / f.with_suffix('.txt').name)
