#!/usr/bin/env python3
"""Make Sense 导出的 COCO JSON → YOLOv8 实例分割格式"""

import json
import os
import shutil
import random

# ===== 配置 =====
COCO_JSON = "/home/jetson/Desktop/vision/labels_my-project-name_2026-07-29-04-05-48(1).json"
DATASET_DIR = "/home/jetson/Desktop/vision/dataset"
SCREENSHOTS_DIR = "/home/jetson/Desktop/vision/screenshots"
TRAIN_RATIO = 0.8   # 80% 训练, 20% 验证

# Make Sense 类别 ID → YOLO class_id（从 0 开始）
CATEGORY_MAPPING = {1: 0}   # "1" → class 0 (扑克牌碎片)
CLASS_NAMES = {0: "fragment"}

# ===== 创建目录结构 =====
for split in ["train", "val"]:
    os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

# ===== 解析 COCO JSON =====
with open(COCO_JSON, 'r') as f:
    data = json.load(f)

# 图片信息: id → (file_name, width, height)
img_info = {img['id']: (img['file_name'], img['width'], img['height']) for img in data['images']}

# 按图片 ID 分组标注
anns_by_image = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in anns_by_image:
        anns_by_image[img_id] = []
    anns_by_image[img_id].append(ann)

# ===== 划分 train/val =====
all_img_ids = list(img_info.keys())
random.seed(42)
random.shuffle(all_img_ids)
split_idx = int(len(all_img_ids) * TRAIN_RATIO)
train_ids = set(all_img_ids[:split_idx])
val_ids = set(all_img_ids[split_idx:])

print(f"总图片: {len(all_img_ids)}")
print(f"训练集: {len(train_ids)}  验证集: {len(val_ids)}")
print(f"总标注: {len(data['annotations'])}")

# ===== 生成 YOLO 标签文件 =====
train_count = 0
val_count = 0
skipped = 0

for img_id in sorted(img_info.keys()):
    filename, img_w, img_h = img_info[img_id]
    basename = os.path.splitext(filename)[0]

    # 确定 split
    if img_id in train_ids:
        split = "train"
    else:
        split = "val"

    # 复制图片
    src_img = os.path.join(SCREENSHOTS_DIR, filename)
    dst_img = os.path.join(DATASET_DIR, "images", split, filename)
    if os.path.exists(src_img):
        shutil.copy2(src_img, dst_img)
    else:
        print(f"⚠ 图片不存在，跳过: {src_img}")
        skipped += 1
        continue

    # 生成标签
    txt_path = os.path.join(DATASET_DIR, "labels", split, basename + '.txt')
    with open(txt_path, 'w') as f:
        for ann in anns_by_image.get(img_id, []):
            coco_cat_id = ann['category_id']
            yolo_class_id = CATEGORY_MAPPING.get(coco_cat_id, coco_cat_id - 1)

            seg = ann['segmentation']
            # COCO segmentation 格式: [[x1,y1,x2,y2,...]] 或 [[poly1],[poly2],...]
            if isinstance(seg[0], list):
                polygon = seg[0]  # 取外轮廓
            else:
                polygon = seg

            # 归一化坐标
            norm_points = []
            for i in range(0, len(polygon), 2):
                nx = polygon[i] / img_w
                ny = polygon[i + 1] / img_h
                norm_points.append(f"{nx:.6f}")
                norm_points.append(f"{ny:.6f}")

            f.write(f"{yolo_class_id} " + " ".join(norm_points) + "\n")

    if split == "train":
        train_count += 1
    else:
        val_count += 1

print(f"✓ 训练集: {train_count} 张图片")
print(f"✓ 验证集: {val_count} 张图片")
if skipped:
    print(f"⚠ 跳过 {skipped} 张（源图片缺失）")

# ===== 生成 dataset.yaml =====
yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
yaml_content = f"""# YOLOv8 实例分割数据集配置
path: {DATASET_DIR}
train: images/train
val: images/val

# 类别
names:
  0: fragment
"""

with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\n✓ dataset.yaml 已生成: {yaml_path}")
print(f"✓ 数据集目录结构:")
print(f"   {DATASET_DIR}/")
print(f"   ├── images/")
print(f"   │   ├── train/  ({train_count} 张)")
print(f"   │   └── val/    ({val_count} 张)")
print(f"   ├── labels/")
print(f"   │   ├── train/  ({train_count} 个)")
print(f"   │   └── val/    ({val_count} 个)")
print(f"   └── dataset.yaml")

# ===== 验证输出 =====
print(f"\n===== 格式抽查 =====")
sample_txt = os.path.join(DATASET_DIR, "labels", "train")
txt_files = sorted([f for f in os.listdir(sample_txt) if f.endswith('.txt')])
if txt_files:
    sample = txt_files[0]
    with open(os.path.join(sample_txt, sample), 'r') as f:
        content = f.read().strip()
    print(f"文件: labels/train/{sample}")
    print(f"内容: {content[:200]}...")
    print(f"（class_id + {len(content.split()[1:])//2} 个顶点坐标对）")
