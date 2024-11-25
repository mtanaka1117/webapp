from pathlib import Path


def remap_labels(dir, new_category_map):
    """
    ラベルをYOLOの形式に振りなおす（0~27）
    """
    
    for split in ['train', 'val']:
        label_files = (dir / 'original_labels' / split).glob("*.txt")
        for label_file in label_files:
            with open(label_file, "r") as f:
                lines = f.readlines()

            updated_lines = [] # 修正データを格納
            for line in lines:
                components = line.split()
                old_class_id = int(components[0])

                # マッピングにあるカテゴリ ID のみ修正
                if old_class_id in new_category_map:
                    new_class_id = new_category_map[old_class_id]
                    components[0] = str(new_class_id)
                    updated_lines.append(" ".join(components) + "\n")
                else:
                    pass
                    # print(f"Deleted invalid class ID {old_class_id} in {label_file}")

            # 修正された内容を保存
            if updated_lines:
                new_file = dir / 'labels' / split / label_file.name
                with open(new_file, "w") as f:
                    f.writelines(updated_lines)
            else:
                # ラベルがすべて無効ならファイルを削除
                # label_file.unlink()
                label_file.rename(dir / 'unnecessary_labels' / split / label_file.name)
                print(f"Deleted {label_file} due to no valid labels")


dir = Path("/home/srv-admin/webapp/docker/python/yolo/datasets/Objects365")
target_categories = [1, 8, 9, 11, 14, 19, 40, 43, 55, 62, 74, 95, 107, 116, 126, 133, 170, 206, 208, 209, 226, 239, 244, 252, 285, 307, 358, 362]
new_category_map = {old_id-1: new_id for new_id, old_id in enumerate(target_categories)}  # マッピングを作成

remap_labels(dir, new_category_map)
