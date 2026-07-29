#!/usr/bin/env python3
"""YOLOv8 实例分割模型训练脚本 — Jetson Orin (CPU 模式)"""

import os
# 必须在 import torch/ultralytics 之前设置，否则 DataLoader 仍会尝试 CUDA pin_memory
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ultralytics import YOLO

# ===== 配置 =====
DATASET_YAML = "/home/jetson/Desktop/vision/dataset/dataset.yaml"
MODEL_SIZE = "n"  # n=nano, s=small, m=medium (推荐从 nano 开始)
EPOCHS = 100
IMG_SIZE = 640
BATCH = 4          # CPU 上 batch 不宜太大

def main():
    model_name = f"yolov8{MODEL_SIZE}-seg.pt"
    print(f"加载模型: {model_name}")

    model = YOLO(model_name)

    print(f"\n===== 开始训练 (CPU) =====")
    print(f"数据集: {DATASET_YAML}")
    print(f"Epochs: {EPOCHS}  Batch: {BATCH}  ImgSize: {IMG_SIZE}")

    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        # --- 小数据集优化 ---
        patience=20,
        cos_lr=True,
        warmup_epochs=3,
        # --- 数据增强 ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        # --- 保存设置 ---
        save=True,
        save_period=10,
        project="runs",
        name=f"segment_fragment_{MODEL_SIZE}",
        exist_ok=True,
        # --- CPU 模式 ---
        workers=0,         # CPU 训练用 0 worker，避免多进程开销
        device="cpu",
        verbose=True,
    )

    best_pt = f"runs/segment_fragment_{MODEL_SIZE}/weights/best.pt"
    print(f"\n===== 训练完成 =====")
    print(f"最佳模型: {best_pt}")

    print(f"\n===== 验证集评估 =====")
    metrics = model.val()
    print(f"mAP@50:    {metrics.seg.map50:.4f}")
    print(f"mAP@50-95: {metrics.seg.map:.4f}")

if __name__ == "__main__":
    main()
