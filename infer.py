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
from reassemble import masks_from_yolo, reassemble

# ===== 配置 =====
MODEL_PATH = "/home/jetson/Desktop/vision/runs/segment_fragment_n/weights/best.pt"
CAMERA_INDEX = 0
WIDTH, HEIGHT = 640, 480
FPS = 60
CONF_THRESH = 0.5

WINDOW_NAME = "YOLO Seg - GPU"

# ===== 顶点显示配置 =====
VERTEX_RADIUS = 4          # 顶点圆点半径
VERTEX_COLOR = (0, 0, 255)  # 顶点颜色 (红色)
VERTEX_THICKNESS = -1       # 填充圆点
EDGE_COLOR = (0, 255, 255)  # 多边形边颜色 (黄色)
EDGE_THICKNESS = 2
APPROX_EPSILON = 0.02       # 轮廓近似精度（越小顶点越多，越大越简化）

# ===== 几何中心点配置 =====
CENTROID_RADIUS = 3        # 中心点半径
CENTROID_COLOR = (0, 255, 0)  # 中心点颜色 (绿色)
CENTROID_THICKNESS = -1     # 填充圆点


def build_gst_pipeline(cam_idx, width, height, fps):
    return (
        f"v4l2src device=/dev/video{cam_idx} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=1 max-buffers=2"
    )


def draw_contour_polygon_vertices(image, masks_data, class_ids=None):
    """
    从 YOLO mask 数据中提取多边形顶点（支持凹多边形）并绘制到图像上。

    Args:
        image: OpenCV BGR 图像 (会被原地修改)
        masks_data: YOLO results[0].masks 对象
        class_ids: 每个 mask 的类别 ID 列表（可选，用于按类别着色）
    """
    if masks_data is None:
        return

    for _i, mask_tensor in enumerate(masks_data.data):
        # mask_tensor: (H, W) 的 float tensor，值 0~1
        mask = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)

        # 调整 mask 尺寸以匹配图像
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # 二值化
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # 取最大轮廓
        cnt = max(contours, key=cv2.contourArea)

        # 多边形近似（保留凹形边界）
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_EPSILON * peri, True)

        # 顶点数过少则跳过
        if len(approx) < 3:
            continue

        # --- 绘制多边形边 ---
        for j in range(len(approx)):
            pt1 = tuple(approx[j][0])
            pt2 = tuple(approx[(j + 1) % len(approx)][0])
            cv2.line(image, pt1, pt2, EDGE_COLOR, EDGE_THICKNESS)

        # --- 绘制顶点 ---
        for pt in approx:
            cv2.circle(image, tuple(pt[0]), VERTEX_RADIUS, VERTEX_COLOR, VERTEX_THICKNESS)

        # --- 顶点序号（可选） ---
        for idx, pt in enumerate(approx):
            cv2.putText(image, str(idx + 1),
                        (pt[0][0] + 8, pt[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- 几何中心点 ---
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(image, (cx, cy), CENTROID_RADIUS, CENTROID_COLOR, CENTROID_THICKNESS)


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
    print("按 q 退出 | 空格/s 截图 | r 拼接\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    prev_time = time.time()
    fps_smoothed = 0.0

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
        results = model(frame, verbose=False, conf=CONF_THRESH, half=True)

        # --- 绘制 mask（不画框） ---
        last_annotated = results[0].plot(
            masks=True, boxes=False, labels=True,
            line_width=2, font_size=1.2,
        )

        display = last_annotated

        # --- 绘制多边形顶点（支持凹形） ---
        draw_contour_polygon_vertices(display, results[0].masks)

        # --- FPS ---
        now = time.time()
        dt = now - prev_time
        prev_time = now
        instant_fps = 1.0 / dt if dt > 0 else 0.0
        fps_smoothed = 0.1 * instant_fps + 0.9 * fps_smoothed

        # --- HUD 顶栏 ---
        fps_text = f"FPS: {fps_smoothed:.1f}"
        _, th = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        bar_h = th + 16

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
        cv2.putText(display, fps_text, (10, th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
        elif key == ord('r'):
            try:
                masks = masks_from_yolo(results)
                if len(masks) < 2:
                    print(f"需要至少 2 个碎片，当前仅检测到 {len(masks)} 个")
                else:
                    canvas, _ = reassemble(masks)
                    cv2.imshow("Reassembled", canvas)
                    print("拼接完成 — 窗口 'Reassembled'")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"拼接失败: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
