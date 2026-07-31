#!/usr/bin/python3
"""YOLOv8 实时实例分割 — GPU 推理"""

import cv2
import gc
import subprocess
import sys
import tempfile
import time
import os
from datetime import datetime

import numpy as np
import torch
from ultralytics import YOLO
from reassemble import (masks_from_yolo, reassemble, draw_exploded_view,
                        _exploded_layout, create_combined_view,
                        draw_fragments_on_original, force_max_vertices)

# ===== 配置 =====
MODEL_PATH = "/home/jetson/Desktop/vision/runs/segment_fragment_n/weights/best.pt"
CAMERA_INDEX = 0
WIDTH, HEIGHT = 640, 480
FPS = 60
CONF_THRESH = 0.5

WINDOW_NAME = "YOLO Seg - GPU"
COMBINED_WINDOW_NAME = "Combined View"

# 组合图独立窗口: 主进程把图存成 PNG, 由独立进程 view_image.py 弹出
VIEW_PNG_PATH = os.path.join(tempfile.gettempdir(), "vision_combined_view.png")

# ===== 顶点显示配置 =====
VERTEX_RADIUS = 4          # 顶点圆点半径
VERTEX_COLOR = (0, 0, 255)  # 顶点颜色 (红色)
VERTEX_THICKNESS = -1       # 填充圆点
EDGE_COLOR = (0, 255, 255)  # 多边形边颜色 (黄色)
EDGE_THICKNESS = 2
APPROX_EPSILON = 0.04       # 轮廓近似精度（越小顶点越多，越大越简化）

# ===== 几何中心点配置 =====
CENTROID_RADIUS = 3        # 中心点半径
CENTROID_COLOR = (0, 255, 0)  # 中心点颜色 (绿色)
CENTROID_THICKNESS = -1     # 填充圆点

# ===== 顶点时域平滑配置 =====
SMOOTH_ALPHA = 0.3         # EMA 平滑系数 (0~1, 越小越平滑但延迟越大)
MAX_MATCH_DIST = 30        # 帧间顶点/轨迹匹配的最大距离 (像素)
MAX_LOST_FRAMES = 10       # 目标丢失后轨迹保留的帧数

# ===== 单帧流程 (启动识别 / read) 的时域平滑 =====
DETECT_SMOOTH_FRAMES = 10  # detect_first_target 平滑累积的命中帧数
READ_SMOOTH_FRAMES = 10    # read 流程平滑采集的帧数


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


def mask_to_polygon_pts(mask_tensor, image_shape):
    """mask 张量 → 最大轮廓的多边形近似顶点列表 (与显示/检测共用).

    与 draw_contour_polygon_vertices 使用同一套提取逻辑: 二值化 →
    最大轮廓 → 多边形近似 (保留凹形边界). 顶点数不足 3 时返回 None.
    """
    mask = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    if mask.shape != image_shape[:2]:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, APPROX_EPSILON * peri, True)
    if len(approx) < 3:
        return None
    return force_max_vertices([tuple(p[0]) for p in approx])


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
        raw_pts = mask_to_polygon_pts(mask_tensor, image.shape)
        if raw_pts is None:
            continue

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


def load_model(path=MODEL_PATH):
    """加载 YOLO 模型到 GPU (无 GPU 时回退 CPU)."""
    print(f"加载模型: {path}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    model = YOLO(path)
    model.to(device)
    return model


def open_camera(cam_idx=CAMERA_INDEX, width=WIDTH, height=HEIGHT, fps=FPS):
    """打开摄像头: GStreamer 管道优先, 失败回退默认后端 (V4L2), 再失败回退 /dev/video1.

    注: 部分环境 OpenCV 未编译 GStreamer 支持, 此时默认后端是唯一可用路径
    (默认后端下强制 MJPG + 目标分辨率以保持帧率).

    Returns:
        VideoCapture; 都打不开时返回 None.
    """
    for idx in (cam_idx, 1):
        if idx != cam_idx:
            print(f"/dev/video{cam_idx} 打不开，尝试 /dev/video1 ...")
        gst_pipe = build_gst_pipeline(idx, width, height, fps)
        print(f"[GStreamer] {gst_pipe}")
        cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print(f"GStreamer 打不开，尝试默认后端 /dev/video{idx} ...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if cap.isOpened():
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"摄像头: /dev/video{idx}  {actual_w}x{actual_h}")
            return cap
    print("无法打开摄像头。")
    return None


def detect_first_target(model, cap, max_frames=None,
                        smooth_frames=DETECT_SMOOTH_FRAMES):
    """循环推理, 检测到目标时返回**时域平滑后**的中心点像素坐标与类别名.

    中心点 = 面积最大的 mask 轮廓多边形近似的几何中心 (与显示逻辑一致,
    见 mask_to_polygon_pts / polygon_centroid); 多边形顶点先经
    VertexSmoother 时域平滑 (连续累积 smooth_frames 帧命中) 再算质心,
    抑制单帧分割噪声引起的坐标抖动.

    Args:
        model: 已加载的 YOLO 模型 (load_model 返回值)
        cap:   已打开的 VideoCapture (open_camera 返回值)
        max_frames: 最多读取帧数, None 表示无限等待
        smooth_frames: 平滑累积的命中帧数, 达到后返回平滑质心
                       (目标丢失后轨迹保留 MAX_LOST_FRAMES 帧, 期间恢复
                       继续累积; 新轨迹则重新累积)

    Returns:
        (cx, cy, class_name); 未检测到 (或 max_frames 耗尽 / 读帧失败) 时返回 None.
    """
    smoother = VertexSmoother()
    hit = 0
    cls = "unknown"
    frames = 0
    while max_frames is None or frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败")
            return None
        results = model(frame, verbose=False, conf=CONF_THRESH, half=True)
        masks = results[0].masks
        if masks is not None and len(masks) > 0:
            # 类别 ID 在 boxes.cls 上 (与 masks.data 同序对应), Masks 自身无 cls 属性
            boxes = results[0].boxes
            cls_ids = boxes.cls if boxes is not None else None
            best_area = -1.0
            best_pts = None
            for mi, mask_tensor in enumerate(masks.data):
                pts = mask_to_polygon_pts(mask_tensor, frame.shape)
                if pts is None:
                    continue
                area = cv2.contourArea(np.array(pts))
                if area > best_area:
                    best_area = area
                    best_pts = pts
                    if cls_ids is not None and mi < len(cls_ids):
                        cls = results[0].names[int(cls_ids[mi])]
                    else:
                        cls = "unknown"
            if best_pts is not None:
                # 时域平滑: 平滑器新建轨迹时重新累积
                before = len(smoother.tracks)
                smoothed_pts = smoother.update(best_pts)
                if len(smoother.tracks) > before:
                    hit = 0
                hit += 1
                if hit >= smooth_frames:
                    cx, cy = polygon_centroid(smoothed_pts)
                    return cx, cy, cls
        del results
        frames += 1
    return None


def _show_combined_view(combined):
    """把三合一组合图存为 PNG, 用独立进程弹出窗口 (单次弹出, 可单独关闭).

    窗口由独立的 view_image.py 进程持有: 关闭窗口 (q/Esc/点 X) 或该
    子进程本身崩溃, 都不影响主进程; 主进程保存后立即返回, 不阻塞.
    """
    if not cv2.imwrite(VIEW_PNG_PATH, combined):
        print(f"⚠ 保存组合图失败: {VIEW_PNG_PATH}")
        return
    viewer = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "view_image.py")
    print("✓ 拼接完成 — 组合图已用独立窗口弹出 "
          f"'{COMBINED_WINDOW_NAME}' (爆炸图 | 实物图 | 装配图)")
    print(f"  按 q / Esc 或点击窗口 X 关闭, 不影响主进程")
    print(f"  图片文件: {VIEW_PNG_PATH}")
    subprocess.Popen([sys.executable, viewer, VIEW_PNG_PATH,
                      COMBINED_WINDOW_NAME])


def _smooth_fragments(model, cap, frames=READ_SMOOTH_FRAMES, min_frags=2):
    """连续采集 frames 帧, 对每个碎片多边形顶点做时域平滑.

    碎片身份按质心最近邻跨帧一对一匹配 (与 VertexSmoother 同思路),
    每个碎片一个平滑器; 碎片不足 min_frags 个的帧跳过 (轨迹由平滑器
    内部按 MAX_LOST_FRAMES 保留, 期间恢复可继续累积).

    Returns:
        (last_frame, [平滑多边形...], [平滑质心...]) — 最后一次
        有效帧的结果; 读帧失败返回 None.
    """
    smoothers = []     # 每个碎片: {"sm": VertexSmoother, "cx", "cy", "used"}
    last = None        # (frame, polygons, centroids)
    max_d2 = MAX_MATCH_DIST ** 2
    for _ in range(frames):
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败")
            return None
        results = model(frame, verbose=False, conf=CONF_THRESH, half=True)
        polys = masks_from_yolo(results)
        del results
        if len(polys) < min_frags:
            continue

        for sm in smoothers:
            sm["used"] = False
        matched = []
        for k, pts in enumerate(polys):
            raw_cent = polygon_centroid(pts)
            best_sm, best_d2 = None, max_d2
            for sm in smoothers:
                if sm["used"]:
                    continue
                d2 = (raw_cent[0] - sm["cx"]) ** 2 + (raw_cent[1] - sm["cy"]) ** 2
                if d2 < best_d2:
                    best_d2, best_sm = d2, sm
            if best_sm is not None:
                best_sm["used"] = True
                smoothed = best_sm["sm"].update(pts)
                cent = polygon_centroid(smoothed)
                best_sm["cx"], best_sm["cy"] = cent
            else:
                sm_new = {"sm": VertexSmoother(), "used": True,
                          "cx": raw_cent[0], "cy": raw_cent[1]}
                smoothed = sm_new["sm"].update(pts)
                cent = polygon_centroid(smoothed)
                sm_new["cx"], sm_new["cy"] = cent
                smoothers.append(sm_new)
            matched.append((smoothed, cent))
        # 清理轨迹已超期 (tracks 清空) 的平滑器, 避免残留质心误匹配
        smoothers = [sm for sm in smoothers if sm["sm"].tracks]
        last = (frame, [p for p, _ in matched], [c for _, c in matched])
    return last


def run_read_pipeline(model, cap, show_view=True,
                      smooth_frames=READ_SMOOTH_FRAMES):
    """跑一遍完整装配流程 (等价于主循环按 R 键) 并输出每个碎片的坐标数据.

    连续采集 smooth_frames 帧 → 碎片顶点跨帧时域平滑 → 拼接 → 爆炸图 →
    三合一组合图, 再按原始图像碎片编号返回:

      原始坐标   = 摄像头画面中**平滑后**碎片多边形的几何中心 (像素)
      爆炸图坐标 = 爆炸图 (640x480 画布) 中该碎片位置的几何中心 (像素)
      旋转角     = 拼接对齐时相对原始位姿的旋转角 (°, 顺时针为正、逆时针为负)

    Args:
        model: 已加载的 YOLO 模型 (load_model 返回值)
        cap:   已打开的 VideoCapture (open_camera 返回值)
        show_view: 是否用独立进程弹出三合一组合窗口 (可单独关闭,
                   不影响主进程)
        smooth_frames: 平滑采集帧数 (碎片多边形经时域平滑后再拼接)

    Returns:
        按原始编号 (idx) 排序的列表, 每项 dict:
            {"idx": int, "orig": (cx, cy), "exploded": (cx, cy),
             "rot_deg": float}
        失败 (读帧失败 / 碎片不足 / 拼接失败) 返回 None, 原因已打印.
    """
    smoothed = _smooth_fragments(model, cap, frames=smooth_frames)
    if smoothed is None:
        print("read: 平滑采集帧读取失败")
        return None
    frame, masks, orig_centroids = smoothed
    try:
        if len(masks) < 2:
            print(f"需要至少 2 个碎片，当前仅检测到 {len(masks)} 个")
            return None

        canvas, _, display_frags, _ = reassemble(masks)

        # 爆炸图
        exploded = draw_exploded_view(display_frags)
        positions, offset = _exploded_layout(display_frags)

        # 原始帧上标注碎片
        fragments_img = draw_fragments_on_original(frame, masks)

        # 三合一组合图：爆炸图 | 实物图 | 装配图 (独立进程弹出, 可单独关闭)
        combined = create_combined_view(exploded, fragments_img, canvas)
        if show_view:
            _show_combined_view(combined)

        # ---- 每个碎片的原始坐标 / 爆炸图坐标 / 旋转角 ----
        data = []
        for i, (orig_idx, _poly, rot_deg) in enumerate(display_frags):
            frag_s = (positions[i] + offset).astype(np.int32)   # 与绘制一致
            M = cv2.moments(frag_s.astype(np.float32))
            if M['m00'] > 0:
                ex_c = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            else:
                ex_c = (int(frag_s[:, 0].mean()), int(frag_s[:, 1].mean()))
            data.append({"idx": orig_idx,
                         "orig": orig_centroids[orig_idx],
                         "exploded": ex_c,
                         "rot_deg": float(rot_deg)})
        data.sort(key=lambda d: d["idx"])   # 按原始图像编号 #1 #2 #3 ... 输出
        return data
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"拼接失败: {e}")
        return None


def main():
    model = load_model()

    # --- 打开摄像头 ---
    cap = open_camera()
    if cap is None:
        sys.exit(1)
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

                    # 三合一组合窗口：爆炸图 | 实物图 | 装配图
                    combined = create_combined_view(exploded, fragments_img, canvas)
                    cv2.namedWindow(COMBINED_WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(COMBINED_WINDOW_NAME,
                                     combined.shape[1], combined.shape[0])
                    cv2.imshow(COMBINED_WINDOW_NAME, combined)

                    print("拼接完成 — 组合窗口 'Combined View' (爆炸图 | 实物图 | 装配图)")
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
