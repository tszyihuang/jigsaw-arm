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

import math
import numpy as np
import cv2
import itertools
import time


# ============================================================
#  几何工具
# ============================================================

def _to_float32(pts):
    """将 (N, 1, 2) 或 (N, 2) 顶点转为 (N, 2) float32。"""
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    return arr


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


def _project_onto_segment(pt, a, b):
    """将 pt 投影到线段 ab 上，返回最近点。"""
    ab = b - a
    t = np.dot(pt - a, ab) / np.dot(ab, ab)
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab


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

    len_f = np.linalg.norm(vf)
    len_t = np.linalg.norm(vt)
    if len_f < 1e-9 or len_t < 1e-9:
        # 零长度边 → 返回单位矩阵，避免 NaN
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]], dtype=np.float32)

    nf = vf / len_f
    nt = vt / len_t

    # 方向相反（切割边面对面）
    target = -nt
    cos_t = np.dot(nf, target)
    sin_t = nf[0] * target[1] - nf[1] * target[0]  # 2D cross product (z-component)

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

def _quadrilateral_angles(vertices):
    """返回四边形的四个内角列表 [angle_deg, ...]，用于诊断。非四边形返回 None。"""
    if len(vertices) != 4:
        return None

    pts = _to_float32(vertices)
    angles = []

    for i in range(4):
        prev = pts[(i - 1) % 4]
        curr = pts[i]
        nxt = pts[(i + 1) % 4]

        v1 = prev - curr
        v2 = nxt - curr
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            return None

        cos_a = np.dot(v1, v2) / (n1 * n2)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cos_a)))
        angles.append(angle)

    return angles


def _check_quadrilateral_angles(vertices, angle_tolerance=10.0):
    """
    检测四边形的四个内角是否都在 90° ± angle_tolerance 范围内。
    用于判断拼接结果是否为合法矩形。
    """
    angles = _quadrilateral_angles(vertices)
    if angles is None:
        return False

    for a in angles:
        if abs(a - 90.0) > angle_tolerance:
            return False

    return True


def merge_collinear_edges(hull, angle_threshold=170.0):
    """
    融并冗余顶点：遍历每个顶点，计算其所接两条邻边的夹角。
    - 夹角 >= angle_threshold → 两条边几乎共线（接近 180°），融并该顶点
    - 夹角 <  angle_threshold → 形成明显拐角，保留该顶点
    """
    if len(hull) <= 3:
        return hull

    n = len(hull)
    drop = [False] * n

    for i in range(n):
        prev = hull[(i - 1) % n]
        curr = hull[i]
        nxt = hull[(i + 1) % n]

        # 进入 curr 的边方向（指向 curr）
        v1 = curr - prev
        # 离开 curr 的边方向（背离 curr）
        v2 = nxt - curr
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1.0 or n2 < 1.0:
            drop[i] = True
            continue

        # v1 指向 curr, v2 背离 curr，直行时二者同向
        # → angle_between ≈ 0° → corner_angle ≈ 180°（平直）
        cos_a = np.dot(v1, v2) / (n1 * n2)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angle_between = np.degrees(np.arccos(cos_a))  # 0°=同向, 180°=反向
        corner_angle = 180.0 - angle_between           # 180°=平直, 0°=急弯

        if corner_angle >= angle_threshold:
            drop[i] = True

    kept = hull[~np.array(drop)]
    return kept if len(kept) >= 3 else hull


def merge_polygons(fixed, moving_aligned, ia_fixed, ib_moving):
    """
    融合两个对齐后的多边形（保留凹形，不走凸包）。

    交叉配对：
      fixed[idx_f1] ≈ moving[idx_m2]  →  接合点 v1
      fixed[idx_f2] ≈ moving[idx_m1]  →  接合点 v2

    融合后按顶点顺序绕行：
      v1 → moving[idx_m2+1] → ... → moving[idx_m1-1] → v2
         → fixed[idx_f2+1] → ... → fixed[idx_f1-1] → 回到 v1

    步骤：
      1. 取中点融合成 v1, v2
      2. 按顺序串联顶点（排除贴合边端点）
      3. 简化（approxPolyDP）
      4. 角度判断：邻边夹角 >= 178° → 融并该顶点
    """
    fixed_f = _to_float32(fixed)
    moving_f = _to_float32(moving_aligned)
    n_f, n_m = len(fixed_f), len(moving_f)

    idx_f1 = ia_fixed
    idx_f2 = (ia_fixed + 1) % n_f
    idx_m1 = ib_moving
    idx_m2 = (ib_moving + 1) % n_m

    # ★ 步骤 1: 将 fixed 端点投影到 moving 的贴合边上，消除中点近似引入的折角
    #    fixed[idx_f1] 对应 moving[idx_m2]→moving[idx_m1] 这条边
    #    fixed[idx_f2] 对应 moving[idx_m2]→moving[idx_m1] 这条边
    v1 = _project_onto_segment(fixed_f[idx_f1], moving_f[idx_m1], moving_f[idx_m2])
    v2 = _project_onto_segment(fixed_f[idx_f2], moving_f[idx_m1], moving_f[idx_m2])

    # ★ 步骤 2: 按顶点顺序绕行，排除贴合边的四个端点
    ordered_pts = [v1]

    # 从 moving 的 idx_m2+1 走到 idx_m1-1（沿 moving 多边形方向，跳过贴合边）
    i = (idx_m2 + 1) % n_m
    while i != idx_m1:
        ordered_pts.append(moving_f[i])
        i = (i + 1) % n_m

    ordered_pts.append(v2)

    # 从 fixed 的 idx_f2+1 走到 idx_f1-1（沿 fixed 多边形方向，跳过贴合边）
    i = (idx_f2 + 1) % n_f
    while i != idx_f1:
        ordered_pts.append(fixed_f[i])
        i = (i + 1) % n_f

    merged = np.array(ordered_pts, dtype=np.float32)

    # ★ 步骤 3: 简化轮廓（Douglas-Peucker，保留凹形）
    peri = cv2.arcLength(merged.astype(np.float32), True)
    merged = cv2.approxPolyDP(merged.astype(np.float32), 0.02 * peri, True)
    merged = merged.reshape(-1, 2)

    # ★ 步骤 4: 遍历顶点，判断所接两条邻边的夹角，>= 178° 则融并
    merged = merge_collinear_edges(merged, angle_threshold=170.0)

    return merged  # (N, 2) float32


# ============================================================
#  YOLO → 多边形
# ============================================================

def force_max_vertices(pts, max_vertices=5):
    """
    强制把多边形顶点数缩到 max_vertices（默认 5）。

    顶点数超过 max_vertices 时，迭代删除"最平"的顶点（所接两条邻边夹角
    最接近 0°、几乎在一条直线上的点），每次只删一个、删完重新评估，
    尽量保持图形形状；不超过 max_vertices 时原样返回。
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    pts = [(int(round(x)), int(round(y))) for x, y in pts]

    # 去重：删除相邻重复点，避免零长度边
    cleaned = []
    for p in pts:
        if not cleaned or p != cleaned[-1]:
            cleaned.append(p)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    pts = cleaned

    while len(pts) > max_vertices:
        n = len(pts)
        flattest = 0
        best_angle = 181.0
        for i in range(n):
            px, py = pts[(i - 1) % n]
            cx, cy = pts[i]
            nx_, ny_ = pts[(i + 1) % n]
            v1 = (cx - px, cy - py)
            v2 = (nx_ - cx, ny_ - cy)
            len1 = math.hypot(v1[0], v1[1])
            len2 = math.hypot(v2[0], v2[1])
            if len1 < 1e-6 or len2 < 1e-6:
                flattest = i
                break
            cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
            cos_a = max(-1.0, min(1.0, cos_a))
            angle = math.degrees(math.acos(cos_a))  # 0°=平直, 180°=急弯
            if angle < best_angle:
                best_angle = angle
                flattest = i
        del pts[flattest]

    return pts


def masks_from_yolo(results, num_fragments=4, epsilon=0.04):
    """从 YOLOv8 推理结果中提取多边形顶点列表。

    epsilon: 轮廓近似精度（与 infer.py 显示路径的 APPROX_EPSILON=0.04 一致，
    避免同一 mask 在显示端和拼接端得到不同顶点数）。
    顶点数超过 5 的多边形会被 force_max_vertices 强制缩到 5 个。
    """
    r = results[0]
    if r.masks is None:
        return []

    if r.orig_shape is None:
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
        approx = cv2.approxPolyDP(cnt, epsilon * peri, True)

        if len(approx) >= 3:
            polygons.append(force_max_vertices(approx))

    return polygons


# ============================================================
#  拼接核心
# ============================================================

def _dfs_place(polygons, merged_poly, display_frags, edge_matches, remaining_order,
               area_threshold, target_vertices,
               frag_edges_cache=None, global_best=None):
    """
    DFS 回溯：尝试按 remaining_order 顺序逐个放置碎片。

    每一步尝试所有边对（按得分排序），面积校验通过则递归。
    所有碎片放置完毕后，检查顶点数：
      - == target_vertices → 直接返回（找到解）
      - >  target_vertices → 回溯，尝试其他边对组合

    Returns:
        (merged_poly, display_frags, edge_matches) — 找到的解（最优优先返回 4 顶点）
        None — 该分支无合法解
    """
    if not remaining_order:
        # 所有碎片已放置，检查顶点数
        nv = len(merged_poly)
        if global_best is not None and nv < global_best[0]:
            global_best[0] = nv
        return (merged_poly.copy(),
                [(idx, f.copy(), r) for idx, f, r in display_frags],
                list(edge_matches))

    orig_idx = remaining_order[0]
    rest = remaining_order[1:]
    poly = polygons[orig_idx]
    poly_edges = frag_edges_cache[orig_idx] if frag_edges_cache else polygon_edges(poly)

    # ---- 使用融并后组合体的边来做匹配（非原始碎片边） ----
    merged_edges = polygon_edges(merged_poly)

    # ---- 所有边对评分排序（长度差 >20% 直接跳过，不穷举） ----
    scored_pairs = []
    for mia, (mp1, mp2, m_len, _) in enumerate(merged_edges):
        for ib, (_, _, lb, _) in enumerate(poly_edges):
            diff = abs(m_len - lb)
            max_len_v = max(m_len, lb)

            # 长度差超过 20% → 跳过，不计算仿射
            if max_len_v > 0 and diff / max_len_v > 0.20:
                continue

            score = (1.0 - diff / max_len_v) if max_len_v > 0 else 1.0
            scored_pairs.append((score, mia, ib))

    scored_pairs.sort(key=lambda x: x[0], reverse=True)

    # ---- 依次尝试每条边对（DFS 回溯） ----
    best_result = None   # (merged_poly, display_frags, edge_matches)
    best_verts = float('inf')

    # 预计算已放置碎片的面积（内层循环中不变）
    base_frag_area = sum(float(cv2.contourArea(f.astype(np.float32)))
                         for _, f, _ in display_frags)

    # merged_poly 转为 (N,1,2) 格式供 pointPolygonTest 复用
    merged_contour = merged_poly.reshape(-1, 1, 2)

    for score, mia, ib in scored_pairs:
        # 延迟计算：只对实际尝试的边对计算仿射变换
        mp1, mp2, m_len, _ = merged_edges[mia]
        M = align_matrix(poly_edges[ib], (mp1, mp2, m_len))
        poly_aligned = transform(M, poly)
        # 旋转角（相对原始位姿；屏幕坐标 y 向下：顺时针为负、逆时针为正）
        rot_deg = -math.degrees(math.atan2(M[1, 0], M[0, 0]))

        # ---- 廉价预检：碎片质心在组合体内部 → 重叠过多，面积几乎必不过 ----
        centroid = poly_aligned.mean(axis=0)
        if cv2.pointPolygonTest(merged_contour, tuple(centroid), False) >= 0:
            continue

        poly_aligned_area = float(cv2.contourArea(poly_aligned.astype(np.float32)))

        candidate_merged = merge_polygons(merged_poly, poly_aligned, mia, ib)

        # ---- 面积校验 ----
        merged_area = float(cv2.contourArea(candidate_merged.astype(np.float32)))
        frag_areas = base_frag_area + poly_aligned_area

        area_ratio = (min(merged_area, frag_areas) /
                      max(merged_area, frag_areas) if max(merged_area, frag_areas) > 0 else 0.0)

        if area_ratio < area_threshold:
            continue

        # 递归放置剩余碎片
        new_display = display_frags + [(orig_idx, poly_aligned, rot_deg)]
        new_edge_matches = edge_matches + [(mp1.copy(), mp2.copy())]
        result = _dfs_place(polygons, candidate_merged, new_display, new_edge_matches, rest,
                            area_threshold, target_vertices,
                            frag_edges_cache=frag_edges_cache,
                            global_best=global_best)

        if result is not None:
            mp, df, em = result
            nv = len(mp)
            if nv == target_vertices:
                if _check_quadrilateral_angles(mp):
                    if global_best is not None:
                        global_best[0] = nv
                    return result  # 完美解，立即返回
                else:
                    # 记录为备选，继续回溯尝试其他组合
                    if nv < best_verts:
                        best_verts = nv
                        best_result = result
            else:
                if nv < best_verts:
                    best_verts = nv
                    best_result = result
        # 否则此边对无解，回溯继续尝试下一个

    return best_result


# 爆炸图目标画布尺寸 (宽 × 高)
EXPLODED_VIEW_W = 640
EXPLODED_VIEW_H = 480


def _exploded_layout(fragments, gap=0.5, canvas_size=(EXPLODED_VIEW_W, EXPLODED_VIEW_H)):
    """
    计算爆炸图布局：每个碎片沿「重心 → 碎片质心」方向径向推开。

    与 draw_exploded_view 的绘制逻辑共用同一套几何计算（含超出画布时
    等比缩小推力、布局整体居中），供输出碎片爆炸图坐标等数据使用。

    Args:
        fragments: [(idx, poly, rot_deg), ...] 已对齐的碎片列表
        gap: 缩放系数，越大推得越开（默认 0.5）
        canvas_size: 输出画布尺寸 (宽, 高)

    Returns:
        (positions, offset):
          positions[i] — fragments[i] 对应碎片在爆炸图画布上的顶点坐标
          offset       — 布局整体居中的平移量
    """
    tw, th = canvas_size
    n = len(fragments)
    originals = [f.copy() for _, f, _ in fragments]
    positions = [f.copy() for f in originals]

    # 1. 计算每个碎片的质心 + 整体重心
    frag_centroids = []
    for poly in positions:
        M = cv2.moments(poly.astype(np.float32))
        if M['m00'] > 0:
            frag_centroids.append(np.array([M['m10'] / M['m00'], M['m01'] / M['m00']], dtype=np.float32))
        else:
            frag_centroids.append(poly.mean(axis=0))
    global_center = np.mean(frag_centroids, axis=0)

    # 2. 每个碎片沿径向向外推移，线性：推力 ∝ 距重心距离
    pushes = []
    for i in range(n):
        direction = frag_centroids[i] - global_center
        pushes.append(direction * gap)
        positions[i] = originals[i] + pushes[i]

    # 3. 若布局超出画布 → 等比缩小推力直到完整可见
    #    （只改变爆炸间距，碎片像素大小保持不变）
    avail_w, avail_h = tw - 2, th - 2
    for _ in range(64):
        bb_min = positions[0].min(axis=0)
        bb_max = positions[0].max(axis=0)
        for p in positions[1:]:
            bb_min = np.minimum(bb_min, p.min(axis=0))
            bb_max = np.maximum(bb_max, p.max(axis=0))
        bw, bh = bb_max - bb_min
        if bw <= avail_w and bh <= avail_h:
            break
        s = min(avail_w / max(float(bw), 1.0), avail_h / max(float(bh), 1.0))
        if s >= 1.0:
            break
        for i in range(n):
            pushes[i] = pushes[i] * s
            positions[i] = originals[i] + pushes[i]

    # 4. 布局中心对齐画布中心（纯平移，碎片大小不变）
    bb_min = positions[0].min(axis=0)
    bb_max = positions[0].max(axis=0)
    for p in positions[1:]:
        bb_min = np.minimum(bb_min, p.min(axis=0))
        bb_max = np.maximum(bb_max, p.max(axis=0))
    offset = np.array([(tw - (bb_max[0] - bb_min[0])) / 2 - bb_min[0],
                       (th - (bb_max[1] - bb_min[1])) / 2 - bb_min[1]],
                      dtype=np.float32)

    return positions, offset

# 几何中心点绘制配置
CENTROID_RADIUS = 4           # 中心点半径
CENTROID_COLOR = (0, 255, 0)  # 中心点颜色 (绿色)
CENTROID_THICKNESS = -1       # 填充圆点


def draw_exploded_view(fragments, gap=0.5, canvas_size=(EXPLODED_VIEW_W, EXPLODED_VIEW_H)):
    """
    绘制爆炸图：重心径向推开。

    碎片保持实际像素大小不变（与摄像头画面 1:1，所占像素一致），
    每个碎片沿「重心 → 碎片质心」方向向外推移，推力与距重心距离成正比。
    若推开后的布局超出画布，自动等比缩小推力（只改变爆炸间距，不改变
    碎片大小），布局整体居中绘制在固定尺寸画布上，并标出每个碎片的
    几何中心坐标。

    Args:
        fragments: [(idx, poly, rot_deg), ...] 已对齐的碎片列表
                   (rot_deg 为相对原始位姿的旋转角, 顺时针为负、逆时针为正)
        gap: 缩放系数，越大推得越开（默认 0.3）
        canvas_size: 输出画布尺寸 (宽, 高)

    Returns:
        canvas: 爆炸图图像（固定 canvas_size，碎片 1:1 像素）
    """
    if len(fragments) < 2:
        return draw_result(fragments)

    tw, th = canvas_size
    positions, offset = _exploded_layout(fragments, gap, canvas_size)

    # ---- 绘制 ----
    canvas = np.full((th, tw, 3), 30, dtype=np.uint8)

    for i, (orig_idx, _frag, rot_deg) in enumerate(fragments):
        color = FRAGMENT_COLORS[i % len(FRAGMENT_COLORS)]
        thickness = 3 if i == 0 else 2

        frag_s = (positions[i] + offset).astype(np.int32)

        # 半透明填充
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [frag_s], color)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        # 轮廓
        cv2.polylines(canvas, [frag_s], True, color, thickness)

        # 编号
        M = cv2.moments(frag_s.astype(np.float32))
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(canvas, f"#{orig_idx + 1}", (cx - 15, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 几何中心点 + 坐标文字
            cv2.circle(canvas, (cx, cy), CENTROID_RADIUS, CENTROID_COLOR, CENTROID_THICKNESS)
            cv2.putText(canvas, f"({cx}, {cy})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CENTROID_COLOR, 1)

            # 旋转角标识（相对原图形；顺时针为负、逆时针为正；固定碎片为 0°）
            rot_text = f"{rot_deg:+.0f}" if rot_deg else "0"
            rot_size, _ = cv2.getTextSize(rot_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(canvas, rot_text, (cx - 15, cy + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            # 度符号 "°" 手工画小圆点（Hershey 字体无此字符, putText 会画成 "?"）
            cv2.circle(canvas, (cx - 15 + rot_size[0] + 2, cy + 28 - rot_size[1] + 2),
                       2, (0, 255, 255), 1)

    return canvas


def reassemble(masks, area_threshold=0.92, target_vertices=4):
    """
    碎片拼接主逻辑（DFS 回溯 + 顶点数校验）。

    1. 枚举每个碎片作为固定起始碎片
    2. 枚举剩余碎片的每种放置顺序
    3. 每一步尝试所有边对组合（按长度匹配度排序）
    4. 面积校验通过则递归，不通过则回溯尝试下一条边对
    5. 所有碎片放置完毕后：
       - 顶点数 == target_vertices → 输出结果 ✓
       - 顶点数 >  target_vertices → 回溯，尝试其他组合
    6. 如果遍历完所有组合仍无 target_vertices 的解，返回最接近的结果

    Args:
        area_threshold: 面积比值阈值，低于此值视为非法融合
        target_vertices: 目标顶点数（默认 4，即四边形）

    Returns:
        (canvas, [merged_poly], display_frags, edge_matches)
    """
    if len(masks) < 2:
        raise ValueError(f"需要至少 2 个碎片，当前仅有 {len(masks)} 个")

    start_time = time.time()

    polygons = [_to_float32(p) for p in masks]
    n = len(polygons)

    # 预计算所有碎片的边列表（避免 DFS 中重复调用 np.linalg.norm）
    frag_edges_cache = [polygon_edges(p) for p in polygons]

    best_result = None   # (merged_poly, display_frags)
    best_verts = float('inf')
    global_best = [float('inf')]  # 跨 combo 共享最优顶点数（mutable 容器）

    # ---- 枚举所有固定碎片 + 放置顺序 ----
    for fixed_idx in range(n):
        remaining_ids = [i for i in range(n) if i != fixed_idx]

        for perm in itertools.permutations(remaining_ids):
            # 剪枝：即便剩余碎片全完美融并（-2 顶点/个），终点数仍不优于已知最优 → 跳过
            remaining_count = len(perm)
            if (len(polygons[fixed_idx]) - 2 * remaining_count) >= global_best[0]:
                continue

            merged_poly = polygons[fixed_idx].copy()
            display_frags = [(fixed_idx, polygons[fixed_idx].copy(), 0.0)]
            edge_matches = [None]  # 固定碎片没有匹配边

            result = _dfs_place(polygons, merged_poly, display_frags, edge_matches, perm,
                                area_threshold, target_vertices,
                                frag_edges_cache=frag_edges_cache,
                                global_best=global_best)

            if result is not None:
                mp, df, em = result
                nv = len(mp)
                if nv == target_vertices and _check_quadrilateral_angles(mp):
                    elapsed = time.time() - start_time
                    print(f"✓ 拼接成功: {nv} 顶点 (四角均 ≈90°), 耗时 {elapsed:.1f}s")
                    canvas = draw_result(df, mp)
                    return canvas, [mp], df, em
                if nv < best_verts:
                    best_verts = nv
                    best_result = result
                    if nv < global_best[0]:
                        global_best[0] = nv

    # ---- 回退：返回最接近目标顶点数的结果 ----
    if best_result is not None:
        mp, df, em = best_result
        elapsed = time.time() - start_time
        nv = len(mp)

        if nv == target_vertices:
            if _check_quadrilateral_angles(mp):
                print(f"✓ 拼接成功: {nv} 顶点 (四角均 ≈90°), 耗时 {elapsed:.1f}s")
            else:
                angles = _quadrilateral_angles(mp)
                angle_str = ", ".join(f"{a:.0f}°" for a in angles) if angles else "N/A"
                print(f"✗ 部分成功: {nv} 顶点但角度偏离 (内角: {angle_str}), 耗时 {elapsed:.1f}s")
        else:
            print(f"△ 部分成功: {nv} 顶点 (未达目标 {target_vertices}), 耗时 {elapsed:.1f}s")

        canvas = draw_result(df, mp)
        return canvas, [mp], df, em

    elapsed = time.time() - start_time
    raise ValueError(f"✗ 拼接失败: 无合法方案, 耗时 {elapsed:.1f}s")


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

# 组合视图面板尺寸（与摄像头分辨率一致，保证 1:1）
COMBINED_PANEL_W = 640
COMBINED_PANEL_H = 480


def draw_result(fragments, merged=None):
    """
    绘制拼接示意图。

    - 彩色半透明填充 + 轮廓 = 各碎片（对齐后的位置）
    - 白色粗轮廓 = 融合后的最终多边形（如果提供）
    - #1 #2 ... 标注选择顺序
    """
    all_pts_list = [f for _, f, _ in fragments]
    if merged is not None:
        all_pts_list.append(merged)
    all_pts = np.vstack(all_pts_list).astype(np.int32)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    margin = 60
    w = x_max - x_min + 2 * margin
    h = y_max - y_min + 2 * margin
    offset = np.array([-x_min + margin, -y_min + margin])

    canvas = np.full((h, w, 3), 30, dtype=np.uint8)

    # --- 各碎片（半透明） ---
    for i, (_, frag, _) in enumerate(fragments):
        color = FRAGMENT_COLORS[i % len(FRAGMENT_COLORS)]
        thickness = 3 if i == 0 else 2   # 第1个碎片（固定）用粗轮廓

        frag_s = (frag + offset).astype(np.int32)

        # 半透明填充
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [frag_s], color)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        # 轮廓
        cv2.polylines(canvas, [frag_s], True, color, thickness)

        # 选择顺序标注（几何中心）
        M = cv2.moments(frag_s.astype(np.float32))
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(canvas, f"#{i + 1}", (cx - 15, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- 融合后多边形（白色粗轮廓） ---
    if merged is not None:
        merged_s = (merged + offset).astype(np.int32)
        cv2.polylines(canvas, [merged_s], True, (255, 255, 255), 3)

    return canvas


def draw_fragments_on_original(image, polygons):
    """
    在原始图像上绘制检测到的碎片轮廓、顶点和编号。

    Args:
        image: 原始 BGR 图像
        polygons: 多边形列表 [(N,1,2) 或 (N,2)]

    Returns:
        带标注的图像副本
    """
    result = image.copy()
    for i, poly in enumerate(polygons):
        color = FRAGMENT_COLORS[i % len(FRAGMENT_COLORS)]
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)

        # 半透明填充
        overlay = result.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.25, result, 0.75, 0, result)

        # 轮廓
        cv2.polylines(result, [pts], True, color, 3)

        # 顶点
        for pt in pts:
            cv2.circle(result, tuple(pt[0]), 4, (0, 0, 255), -1)

        # 编号
        M = cv2.moments(pts.astype(np.float32))
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(result, f"#{i + 1}", (cx - 15, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return result


def create_combined_view(exploded_img, fragment_img, reassembly_img=None):
    """
    创建组合视图：爆炸图 | 实物图 | 装配图 横向排列在一个窗口。

    每个面板固定 COMBINED_PANEL_W x COMBINED_PANEL_H（640x480，与摄像头
    分辨率一致，碎片 1:1），图像等比缩放居中放入面板（不放大，只缩小
    过大的图像）。

    Args:
        exploded_img: 爆炸图 BGR 图像
        fragment_img: 带碎片标注的实物图（摄像头画面）
        reassembly_img: 装配图 BGR 图像（可选，传入则显示三面板）

    Returns:
        组合画布 (H, W, 3) BGR
    """
    panel_w, panel_h = COMBINED_PANEL_W, COMBINED_PANEL_H
    title_h = 40
    divider = 2   # 面板分隔线宽度

    images = [exploded_img, fragment_img]
    titles = ["Exploded View", "Camera / Fragments"]
    if reassembly_img is not None:
        images.append(reassembly_img)
        titles.append("Reassembly")

    n = len(images)
    canvas_w = n * panel_w + (n - 1) * divider
    canvas_h = title_h + panel_h
    canvas = np.full((canvas_h, canvas_w, 3), 35, dtype=np.uint8)

    for i, (img, title) in enumerate(zip(images, titles)):
        x0 = i * (panel_w + divider)
        if img is not None:
            ih, iw = img.shape[:2]
            scale = min(panel_w / iw, panel_h / ih, 1.0)   # 不放大，只缩小
            new_w = max(int(iw * scale), 1)
            new_h = max(int(ih * scale), 1)
            if (new_w, new_h) != (iw, ih):
                img = cv2.resize(img, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
            x = x0 + (panel_w - new_w) // 2
            y = title_h + (panel_h - new_h) // 2
            canvas[y:y + new_h, x:x + new_w] = img

        cv2.putText(canvas, title, (x0 + 20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    # ---- 面板分隔线（画在面板之间的空隙正中间，不覆盖面板内容） ----
    for i in range(1, n):
        x = i * (panel_w + divider) - divider // 2
        cv2.line(canvas, (x, title_h), (x, canvas_h - 1), (80, 80, 80), 1)

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
    if img is None:
        print(f"错误: 无法读取图像 '{sys.argv[1]}'")
        sys.exit(1)
    results = model(img, verbose=False, conf=0.5)

    try:
        polygons = masks_from_yolo(results, num_fragments=4)
        if len(polygons) < 2:
            print(f"仅检测到 {len(polygons)} 个碎片（需要 ≥2）")
            sys.exit(1)

        canvas, _, display_frags, edge_matches = reassemble(polygons)

        # 爆炸图
        exploded = draw_exploded_view(display_frags)

        # 原始图像上标注碎片
        fragments_img = draw_fragments_on_original(img, polygons)

        # 三合一组合窗口：爆炸图 | 实物图 | 装配图
        combined = create_combined_view(exploded, fragments_img, canvas)
        cv2.imshow("Combined View", combined)

        cv2.waitKey(0)
        if not cv2.imwrite("reassembled.jpg", canvas):
            print("警告: 保存 reassembled.jpg 失败（磁盘满或权限不足）")
        else:
            print("已保存: reassembled.jpg")
        if not cv2.imwrite("exploded.jpg", exploded):
            print("警告: 保存 exploded.jpg 失败（磁盘满或权限不足）")
        else:
            print("已保存: exploded.jpg")
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import sys
    import os
    main()
