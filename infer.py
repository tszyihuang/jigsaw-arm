#!/usr/bin/env python3
"""YOLOv8 实时实例分割 — GPU 推理"""

import cv2
import gc
import sys
import time
import os
from datetime import datetime

import numpy as np
import torch
from ultralytics import YOLO
from reassemble import masks_from_yolo, reassemble, draw_exploded_view, create_combined_view, draw_fragments_on_original

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
APPROX_EPSILON = 0.03       # 轮廓近似精度（越小顶点越多，越大越简化）

# ===== 几何中心点配置 =====
CENTROID_RADIUS = 3        # 中心点半径
CENTROID_COLOR = (0, 255, 0)  # 中心点颜色 (绿色)
CENTROID_THICKNESS = -1     # 填充圆点

# ===== 顶点时域平滑配置 =====
SMOOTH_ALPHA = 0.4         # EMA 平滑系数 (0~1, 越小越平滑但延迟越大)
MAX_MATCH_DIST = 30        # 帧间顶点/轨迹匹配的最大距离 (像素)
MAX_LOST_FRAMES = 10       # 目标丢失后轨迹保留的帧数


def build_gst_pipeline(cam_idx, width, height, fps):
    return (
        f"v4l2src device=/dev/video{cam_idx} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=1 max-buffers=2"
    )


def polygon_centroid(pts):
    """多边形几何中心（鞋带公式质心）"""
    n = len(pts)
    if n < 3:
        return (int(round(np.mean([p[0] for p in pts]))),
                int(round(np.mean([p[1] for p in pts]))))
    area = 0.0
    cx = cy = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-6:
        return (int(round(np.mean([p[0] for p in pts]))),
                int(round(np.mean([p[1] for p in pts]))))
    return (int(round(cx / (6.0 * area))),
            int(round(cy / (6.0 * area))))


class VertexSmoother:
    """
    顶点时域平滑器：跨帧最近邻匹配 + 指数移动平均 (EMA)。

    每帧先用多边形质心匹配目标轨迹，再对顶点做一对一最近邻匹配，
    匹配上的顶点用 EMA 低通滤波，新出现的顶点直接初始化，
    从而抑制 mask 分割噪声引起的顶点抖动。
    """

    def __init__(self, alpha=SMOOTH_ALPHA, max_match_dist=MAX_MATCH_DIST,
                 max_lost_frames=MAX_LOST_FRAMES):
        self.alpha = alpha
        self.max_match_dist = max_match_dist
        self.max_lost_frames = max_lost_frames
        self.tracks = []   # [{pts, cx, cy, age}, ...]

    def update(self, pts):
        """输入当前帧顶点列表，返回平滑后的顶点列表"""
        raw_cent = polygon_centroid(pts)
        # --- 用质心匹配目标轨迹 ---
        best_t, best_d = None, self.max_match_dist ** 2
        for t in self.tracks:
            d2 = (raw_cent[0] - t["cx"]) ** 2 + (raw_cent[1] - t["cy"]) ** 2
            if d2 < best_d:
                best_d, best_t = d2, t
        if best_t is None:
            self.tracks.append({"pts": list(pts), "cx": raw_cent[0],
                                "cy": raw_cent[1], "age": 0})
            return list(pts)
        best_t["age"] = 0
        # --- 顶点一对一最近邻匹配 + EMA ---
        used = set()
        smoothed = []
        for (x, y) in pts:
            best_j, best_d2 = None, self.max_match_dist ** 2
            for j, (ox, oy) in enumerate(best_t["pts"]):
                if j in used:
                    continue
                d2 = (x - ox) ** 2 + (y - oy) ** 2
                if d2 < best_d2:
                    best_j, best_d2 = j, d2
            if best_j is not None:
                used.add(best_j)
                ox, oy = best_t["pts"][best_j]
                smoothed.append((int(round(ox + self.alpha * (x - ox))),
                                 int(round(oy + self.alpha * (y - oy)))))
            else:
                smoothed.append((x, y))
        best_t["pts"] = smoothed
        # 质心锚点同样做 EMA，用于下一帧的轨迹匹配
        best_t["cx"] = int(round(best_t["cx"] + self.alpha * (raw_cent[0] - best_t["cx"])))
        best_t["cy"] = int(round(best_t["cy"] + self.alpha * (raw_cent[1] - best_t["cy"])))
        # --- 清理长期丢失的轨迹 ---
        for t in self.tracks:
            if t is not best_t:
                t["age"] += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_lost_frames]
        return smoothed


def draw_contour_polygon_vertices(image, masks_data, smoother=None, class_ids=None):
    """
    从 YOLO mask 数据中提取多边形顶点（支持凹多边形）并绘制到图像上。

    Args:
        image: OpenCV BGR 图像 (会被原地修改)
        masks_data: YOLO results[0].masks 对象
        smoother: VertexSmoother 实例，传入则对顶点做时域平滑
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

        # 原始顶点 (保留凹形边界)
        raw_pts = [tuple(p[0]) for p in approx]

        # --- 时域平滑 ---
        if smoother is not None:
            pts = smoother.update(raw_pts)
        else:
            pts = raw_pts

        # --- 绘制多边形边 ---
        for j in range(len(pts)):
            pt1 = pts[j]
            pt2 = pts[(j + 1) % len(pts)]
            cv2.line(image, pt1, pt2, EDGE_COLOR, EDGE_THICKNESS)

        # --- 绘制顶点 ---
        for pt in pts:
            cv2.circle(image, pt, VERTEX_RADIUS, VERTEX_COLOR, VERTEX_THICKNESS)

        # --- 顶点序号（可选） ---
        for idx, pt in enumerate(pts):
            cv2.putText(image, str(idx + 1),
                        (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- 几何中心点（由平滑后多边形计算） ---
        cx, cy = polygon_centroid(pts)
        cv2.circle(image, (cx, cy), CENTROID_RADIUS, CENTROID_COLOR, CENTROID_THICKNESS)
        # 中心点坐标文字
        cv2.putText(image, f"({cx}, {cy})", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CENTROID_COLOR, 1)


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
    print("按 q 退出 | 空格/s 截图 | r 拼接 | t 平滑开关\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    smoother = VertexSmoother()   # 顶点时域平滑器
    smoothing_enabled = True
    frame_count = 0   # 用于周期性清理显存缓存池

    prev_time = time.time()
    fps_smoothed = 0.0

    # 预热：跑一次推理让 CUDA JIT 编译完
    print("预热 CUDA...")
    model(cv2.imread(os.path.join(
        "/home/jetson/Desktop/vision/dataset/images/val",
        sorted(os.listdir("/home/jetson/Desktop/vision/dataset/images/val"))[0]
    )), verbose=False, half=True)
    print(f"预热完成，开始实时推理 (GPU 已用 {torch.cuda.memory_allocated() / 2**20:.0f}MiB)\n")

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
        draw_contour_polygon_vertices(display, results[0].masks,
                                      smoother if smoothing_enabled else None)

        # --- FPS ---
        now = time.time()
        dt = now - prev_time
        prev_time = now
        instant_fps = 1.0 / dt if dt > 0 else 0.0
        fps_smoothed = 0.1 * instant_fps + 0.9 * fps_smoothed

        # --- HUD 顶栏 ---
        fps_text = f"FPS: {fps_smoothed:.1f}  SMTH:{'ON' if smoothing_enabled else 'OFF'}"
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
        elif key == ord('t'):
            smoothing_enabled = not smoothing_enabled
            print(f"顶点时域平滑: {'开' if smoothing_enabled else '关'}")
        elif key == ord('r'):
            try:
                masks = masks_from_yolo(results)
                if len(masks) < 2:
                    print(f"需要至少 2 个碎片，当前仅检测到 {len(masks)} 个")
                else:
                    canvas, _, display_frags, edge_matches = reassemble(masks)

                    # 爆炸图
                    exploded = draw_exploded_view(display_frags)

                    # 原始帧上标注碎片
                    fragments_img = draw_fragments_on_original(clean, masks)

                    # A4 横向组合窗口：左 = 爆炸图，右 = 摄像头碎片
                    combined = create_combined_view(exploded, fragments_img)
                    cv2.imshow("Fragments & Exploded", combined)

                    # 装配图独立窗口
                    cv2.imshow("Reassembly", canvas)

                    print("拼接完成 — 窗口 'Reassembly' + 'Fragments & Exploded'")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"拼接失败: {e}")

        # --- 显存管理: 本帧 GPU 张量及时释放, 周期性清缓存池 ---
        del results
        frame_count += 1
        if frame_count % 300 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[Mem] allocated={torch.cuda.memory_allocated() / 2**20:.0f}MiB"
                  f" reserved={torch.cuda.memory_reserved() / 2**20:.0f}MiB")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
