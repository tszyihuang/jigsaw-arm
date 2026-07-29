#!/usr/bin/env python3
"""YOLOv8 实时实例分割 — GPU 推理"""

import cv2
import sys
import time
import os
from datetime import datetime

import numpy as np
import torch
from ultralytics import YOLO

# ===== 配置 =====
MODEL_PATH = "/home/jetson/Desktop/vision/runs/segment_fragment_n/weights/best.pt"
CAMERA_INDEX = 0
WIDTH, HEIGHT = 640, 480
FPS = 60
CONF_THRESH = 0.5

WINDOW_NAME = "YOLO Seg - GPU"


def build_gst_pipeline(cam_idx, width, height, fps):
    return (
        f"v4l2src device=/dev/video{cam_idx} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=1 max-buffers=2"
    )


def main():
    # --- 加载模型到 GPU ---
    print(f"加载模型: {MODEL_PATH}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    model = YOLO(MODEL_PATH)
    model.to(device)

    # --- 打开摄像头 ---
    dev = CAMERA_INDEX
    gst_pipe = build_gst_pipeline(dev, WIDTH, HEIGHT, FPS)
    print(f"[GStreamer] {gst_pipe}")

    cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"/dev/video{dev} 打不开，尝试 /dev/video1 ...")
        dev = 1
        gst_pipe = build_gst_pipeline(dev, WIDTH, HEIGHT, FPS)
        cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("无法打开摄像头。")
            sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头: /dev/video{dev}  {actual_w}x{actual_h}")
    screenshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    print("按 q 退出 | 空格/s 截图 | +/- 调整置信度\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    prev_time = time.time()
    fps_smoothed = 0.0
    conf_thresh = CONF_THRESH

    # 预热：跑一次推理让 CUDA JIT 编译完
    print("预热 CUDA...")
    model(cv2.imread(os.path.join(
        "/home/jetson/Desktop/vision/dataset/images/val",
        sorted(os.listdir("/home/jetson/Desktop/vision/dataset/images/val"))[0]
    )), verbose=False, half=True)
    print("预热完成，开始实时推理\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败")
            break

        clean = frame.copy()

        # --- 半精度推理（FP16，省显存） ---
        results = model(frame, verbose=False, conf=conf_thresh, half=True)

        # --- 绘制 mask（不画框） ---
        last_annotated = results[0].plot(
            masks=True, boxes=False, labels=True,
            line_width=2, font_size=1.2,
        )

        display = last_annotated

        # --- FPS ---
        now = time.time()
        dt = now - prev_time
        prev_time = now
        instant_fps = 1.0 / dt if dt > 0 else 0.0
        fps_smoothed = 0.1 * instant_fps + 0.9 * fps_smoothed

        # --- HUD 顶栏 ---
        fps_text = f"FPS: {fps_smoothed:.1f}"
        conf_text = f"Conf: {conf_thresh:.2f}"
        _, th = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        bar_h = th + 16

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
        cv2.putText(display, fps_text, (10, th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, conf_text, (140, th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, display)

        # --- 按键 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("退出")
            break
        elif key == ord('s') or key == 32:
            filename = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S_%f") + ".jpg"
            filepath = os.path.join(screenshots_dir, filename)
            cv2.imwrite(filepath, clean)
            print(f"已保存: {filepath}")
        elif key == ord('+') or key == ord('='):
            conf_thresh = min(1.0, conf_thresh + 0.05)
            print(f"Conf: {conf_thresh:.2f}")
        elif key == ord('-'):
            conf_thresh = max(0.1, conf_thresh - 0.05)
            print(f"Conf: {conf_thresh:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
