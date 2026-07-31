#!/usr/bin/env python3
"""USB Camera Viewer - GStreamer + MJPG 硬件加速, 60fps, 画面叠加 FPS."""

import cv2
import sys
import time
import os
from datetime import datetime

CAMERA_INDEX = 0          # /dev/video0
WIDTH, HEIGHT = 640, 480
FPS = 60

WINDOW_NAME = "USB Camera (MJPG 60fps)"


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
            print("仍然打不开，请检查摄像头连接。")
            sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头已打开: /dev/video{dev}  {actual_w}x{actual_h} @ 目标 {FPS}fps")
    screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    print("按 'q' 退出, 按 空格 或 's' 截图保存")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    # FPS 平滑统计（滑动平均）
    prev_time = time.time()
    fps_smoothed = 0.0
    alpha = 0.1  # 平滑系数（越小越平滑）

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败")
            break

        # ---- 计算 FPS ----
        now = time.time()
        dt = now - prev_time
        prev_time = now
        instant_fps = 1.0 / dt if dt > 0 else 0.0
        fps_smoothed = alpha * instant_fps + (1 - alpha) * fps_smoothed

        # ---- 在画面左上角叠加 FPS ----
        fps_text = f"FPS: {fps_smoothed:.1f}"
        # 半透明背景条
        (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (12 + tw, 14 + th), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, fps_text, (10, th + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("退出")
            break
        elif key == ord('s') or key == 32:  # 's' 或 空格键
            filename = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S_%f") + ".jpg"
            filepath = os.path.join(screenshot_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"已保存 {filepath}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
