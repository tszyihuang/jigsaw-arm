#!/usr/bin/env python3
"""
碎片拼接

逻辑：
  1. 从 YOLO masks 中检测多边形碎片
  2. 随机挑选一个碎片固定
  3. 遍历固定碎片的每条边，从其他碎片中找长度最接近的边
  4. 将匹配的碎片通过仿射变换对齐到固定边 → 融合
  5. 重复直到所有碎片合并完毕，显示拼接结果
"""

import numpy as np
import cv2
import random


# ============================================================
#  几何工具
# ============================================================

def _to_float32(pts):
    """将 (N, 1, 2) 或 (N, 2) 顶点转为 (N, 2) float32。"""
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    return arr


def _to_approx(pts):
    """将 (N, 2) float32 转回 approxPolyDP 格式 (N, 1, 2) int。"""
    return pts.reshape(-1, 1, 2).astype(np.int32)


def _edge_len(p1, p2):
    return float(np.linalg.norm(np.asarray(p1, dtype=np.float32) -
                                np.asarray(p2, dtype=np.float32)))


def polygon_edges(vertices):
    """
    从多边形顶点提取边列表。
    Returns: [(p1, p2, length, index), ...]  — p1/p2 为 (2,) float32
    """
    pts = _to_float32(vertices)
    edges = []
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        edges.append((p1, p2, _edge_len(p1, p2), i))
    return edges


# ============================================================
#  仿射对齐
# ============================================================

def align_matrix(edge_from, edge_to):
    """
    计算 2x3 仿射矩阵，把 edge_from 对齐到 edge_to。

    方式：两边中点重合、方向相反、共享同一条垂直平分线。
    短边居中落在长边上。
    """
    p1_f, p2_f = edge_from[0], edge_from[1]
    p1_t, p2_t = edge_to[0], edge_to[1]

    mf = (p1_f + p2_f) / 2.0
    mt = (p1_t + p2_t) / 2.0

    vf = p2_f - p1_f
    vt = p2_t - p1_t

    nf = vf / np.linalg.norm(vf)
    nt = vt / np.linalg.norm(vt)

    # 方向相反（切割边面对面）
    target = -nt
    cos_t = np.dot(nf, target)
    sin_t = np.cross(nf, target)

    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]])
    t = mt - R @ mf

    return np.column_stack([R, t])


def transform(M, pts):
    """对 (N, 2) 点集应用 2x3 仿射变换。"""
    R, t = M[:, :2], M[:, 2]
    pts_f = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    return (R @ pts_f.T).T + t


# ============================================================
#  多边形融合
# ============================================================

def merge_collinear_edges(hull, angle_tol=3.0):
    """移除共线相邻边之间的冗余顶点。"""
    if len(hull) <= 3:
        return hull

    n = len(hull)
    drop = [False] * n

    for i in range(n):
        prev = hull[(i - 1) % n]
        curr = hull[i]
        nxt = hull[(i + 1) % n]

        v1 = curr - prev
        v2 = nxt - curr
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1.0 or n2 < 1.0:
            drop[i] = True
            continue

        cos_a = np.dot(v1, v2) / (n1 * n2)
        angle = np.degrees(np.arccos(np.clip(abs(cos_a), 0, 1)))
        if angle < angle_tol:
            drop[i] = True

    kept = hull[~np.array(drop)]
    return kept if len(kept) >= 3 else hull


def merge_polygons(fixed, moving_aligned, ia_fixed, ib_moving):
    """
    融合两个对齐后的多边形。

    端点交叉配对（方向相反）→ 取中点 → 排除贴合边端点 →
    加融合顶点 → 凸包 → 简化 → 去共线。
    """
    fixed_f = _to_float32(fixed)
    moving_f = _to_float32(moving_aligned)
    n_f, n_m = len(fixed_f), len(moving_f)

    idx_f1 = ia_fixed
    idx_f2 = (ia_fixed + 1) % n_f
    idx_m1 = ib_moving
    idx_m2 = (ib_moving + 1) % n_m

    # 交叉配对取中点
    v1 = (fixed_f[idx_f1] + moving_f[idx_m2]) / 2.0
    v2 = (fixed_f[idx_f2] + moving_f[idx_m1]) / 2.0

    pts = [v1, v2]
    for i in range(n_f):
        if i != idx_f1 and i != idx_f2:
            pts.append(fixed_f[i])
    for i in range(n_m):
        if i != idx_m1 and i != idx_m2:
            pts.append(moving_f[i])

    all_pts = np.array(pts, dtype=np.float32)

    merged = cv2.convexHull(all_pts).reshape(-1, 2)
    peri = cv2.arcLength(merged.astype(np.float32), True)
    merged = cv2.approxPolyDP(merged.astype(np.float32), 0.02 * peri, True)
    merged = merged.reshape(-1, 2)
    merged = merge_collinear_edges(merged)

    return merged  # (N, 2) float32


# ============================================================
#  YOLO → 多边形
# ============================================================

def masks_from_yolo(results, num_fragments=4):
    """从 YOLOv8 推理结果中提取多边形顶点列表。"""
    r = results[0]
    if r.masks is None:
        return []

    h, w = r.orig_shape
    polygons = []

    for mask_tensor in r.masks.data[:num_fragments]:
        m = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) >= 3:
            polygons.append(approx)

    return polygons


# ============================================================
#  拼接核心
# ============================================================

def reassemble(masks, canvas_size=(640, 480)):
    """
    碎片拼接主逻辑。

    1. 随机挑选一个碎片固定
    2. 遍历固定碎片的每条边，从其他碎片中找长度最接近的边
    3. 仿射对齐：将碎片旋转/平移到贴合位置（不融合）
    4. 显示所有对齐后的碎片

    Returns:
        (canvas, aligned_fragments): 可视化图像和对齐后的碎片列表
    """
    if len(masks) < 2:
        raise ValueError(f"需要至少 2 个碎片，当前仅有 {len(masks)} 个")

    polygons = [_to_float32(p) for p in masks]
    n = len(polygons)

    # ---- 1. 随机挑选一个固定 ----
    fixed_idx = random.randrange(n)
    fixed_poly = polygons[fixed_idx]
    aligned_frags = [fixed_poly.copy()]  # 第一个是固定碎片

    remaining = [(i, polygons[i]) for i in range(n) if i != fixed_idx]

    print(f"固定碎片: {fixed_idx + 1}  ({len(fixed_poly)} 顶点)")

    # ---- 2. 逐个对齐剩余碎片 ----
    for step, (orig_idx, poly) in enumerate(remaining):
        fixed_edges = polygon_edges(fixed_poly)
        poly_edges = polygon_edges(poly)

        # 找长度最接近的边对
        best_ia, best_ib = 0, 0
        best_diff = float('inf')
        for ia, (_, _, la, _) in enumerate(fixed_edges):
            for ib, (_, _, lb, _) in enumerate(poly_edges):
                diff = abs(la - lb)
                if diff < best_diff:
                    best_diff = diff
                    best_ia, best_ib = ia, ib

        ia, ib = best_ia, best_ib
        la, lb = fixed_edges[ia][2], poly_edges[ib][2]
        print(f"  第 {step+1} 次对齐: 固定边[{ia}]={la:.1f}px  "
              f"←→ 碎片{orig_idx+1}边[{ib}]={lb:.1f}px  (差 {best_diff:.1f}px)")

        # 仿射对齐：碎片边 → 固定边
        M = align_matrix(poly_edges[ib], fixed_edges[ia])
        poly_aligned = transform(M, poly)
        aligned_frags.append(poly_aligned)

    print(f"共对齐 {len(aligned_frags) - 1} 个碎片")

    # ---- 3. 绘制 ----
    canvas = draw_result(aligned_frags)
    return canvas, aligned_frags


# ============================================================
#  可视化
# ============================================================

FRAGMENT_COLORS = [
    (180, 120, 60),   # 蓝
    (60, 160, 210),   # 橙
    (80, 200, 120),   # 绿
    (120, 80, 200),   # 紫
    (200, 160, 80),   # 青
]


def draw_result(fragments):
    """
    绘制拼接示意图。

    - 彩色半透明填充 + 轮廓 = 各碎片（对齐后的位置）
    - 第一个碎片（固定碎片）用黄色粗轮廓突出
    """
    all_pts = np.vstack(fragments).astype(np.int32)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    margin = 60
    w = x_max - x_min + 2 * margin
    h = y_max - y_min + 2 * margin
    offset = np.array([-x_min + margin, -y_min + margin])

    canvas = np.full((h, w, 3), 30, dtype=np.uint8)

    for i, frag in enumerate(fragments):
        color = FRAGMENT_COLORS[i % len(FRAGMENT_COLORS)]
        thickness = 3 if i == 0 else 2   # 固定碎片用粗轮廓

        frag_s = (frag + offset).astype(np.int32)

        # 半透明填充
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [frag_s], color)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        # 轮廓
        cv2.polylines(canvas, [frag_s], True, color, thickness)

    return canvas


# ============================================================
#  命令行入口
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("用法: python reassemble.py <image_path>")
        sys.exit(1)

    from ultralytics import YOLO

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "runs/segment_fragment_n/weights/best.pt")
    model = YOLO(model_path)
    img = cv2.imread(sys.argv[1])
    results = model(img, verbose=False, conf=0.5)

    try:
        polygons = masks_from_yolo(results, num_fragments=4)
        if len(polygons) < 2:
            print(f"仅检测到 {len(polygons)} 个碎片（需要 ≥2）")
            sys.exit(1)

        canvas, _merged = reassemble(polygons)

        cv2.imshow("Reassembly", canvas)
        cv2.waitKey(0)
        cv2.imwrite("reassembled.jpg", canvas)
        print("已保存: reassembled.jpg")
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import sys
    import os
    main()
