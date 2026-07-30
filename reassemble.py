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

def merge_collinear_edges(hull, angle_threshold=175.0):
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
      4. 角度判断：邻边夹角 >= 175° → 融并该顶点
    """
    fixed_f = _to_float32(fixed)
    moving_f = _to_float32(moving_aligned)
    n_f, n_m = len(fixed_f), len(moving_f)

    idx_f1 = ia_fixed
    idx_f2 = (ia_fixed + 1) % n_f
    idx_m1 = ib_moving
    idx_m2 = (ib_moving + 1) % n_m

    # ★ 步骤 1: 无论什么情况，先取中点融合成一个顶点
    v1 = (fixed_f[idx_f1] + moving_f[idx_m2]) / 2.0
    v2 = (fixed_f[idx_f2] + moving_f[idx_m1]) / 2.0

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

    # ★ 步骤 4: 遍历顶点，判断所接两条邻边的夹角，>= 175° 则融并
    merged = merge_collinear_edges(merged, angle_threshold=175.0)

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

def reassemble(masks, canvas_size=(640, 480), area_threshold=0.96):
    """
    碎片拼接主逻辑（迭代组合 + 顶点融合 + 面积校验）。

    1. 随机挑选一个碎片固定
    2. 逐个取剩余碎片，与当前融合体的所有边匹配，按长度差排序
    3. 依次尝试边对：仿射对齐 → 顶点融合 → 面积校验
       - 合法（融合面积 ≈ 碎片面积和）→ 接受，继续下一碎片
       - 非法（面积差太大）→ 加入黑名单，尝试下一组边对
    4. 显示最终结果

    Args:
        area_threshold: 面积比值阈值，低于此值视为非法融合（0~1，越大越严格）

    Returns:
        (canvas, [merged_poly]): 可视化图像和融合后的多边形
    """
    if len(masks) < 2:
        raise ValueError(f"需要至少 2 个碎片，当前仅有 {len(masks)} 个")

    polygons = [_to_float32(p) for p in masks]
    n = len(polygons)

    # ---- 1. 随机挑选一个固定 ----
    fixed_idx = random.randrange(n)
    merged_poly = polygons[fixed_idx].copy()
    display_frags = [(fixed_idx, polygons[fixed_idx].copy())]

    remaining = [(i, polygons[i]) for i in range(n) if i != fixed_idx]

    print(f"固定碎片: {fixed_idx + 1}  ({len(merged_poly)} 顶点)")

    # ---- 2. 逐个匹配 + 对齐 + 融合 ----
    for step, (orig_idx, poly) in enumerate(remaining):
        poly_edges = polygon_edges(poly)

        # ★ 收集所有已放置碎片的真实边（非凸包桥接边）
        # each: (p1, p2, length, local_edge_idx, frag_order)
        group_edges = []
        for frag_order, (_, frag) in enumerate(display_frags):
            for e in polygon_edges(frag):
                group_edges.append((*e, frag_order))

        # 所有边对：按边长匹配度评分
        scored_pairs = []
        for ge in group_edges:
            ge_p1, ge_p2, ge_len, ge_local_idx, ge_frag_order = ge
            for ib, (_, _, lb, _) in enumerate(poly_edges):
                M = align_matrix(poly_edges[ib], (ge_p1, ge_p2, ge_len))
                poly_aligned = transform(M, poly)

                diff = abs(ge_len - lb)
                max_len = max(ge_len, lb)
                score = (1.0 - diff / max_len) if max_len > 0 else 1.0

                scored_pairs.append((score, diff, ge, ib, poly_aligned))

        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        blacklist = set()
        accepted = False

        for rank, (score, diff, ge, ib, poly_aligned) in enumerate(scored_pairs):
            ge_p1, ge_p2, ge_len, ge_local_idx, ge_frag_order = ge
            lb = poly_edges[ib][2]
            pair_key = (ge_frag_order, ge_local_idx, ib)
            if pair_key in blacklist:
                continue

            tag = " ← 备选" if rank > 0 else ""
            print(f"  第 {step+1} 次拼接: "
                  f"已放置碎片{ge_frag_order+1}边[{ge_local_idx}]={ge_len:.1f}px  "
                  f"←→ 碎片{orig_idx+1}边[{ib}]={lb:.1f}px  "
                  f"得分={score:.3f}  (差 {diff:.1f}px){tag}")

            # 顶点融合：用当前融合体作为基底
            # 找 merged_poly 中与匹配边最接近的边索引
            merged_edges = polygon_edges(merged_poly)
            best_mia = 0
            best_mid_dist = float('inf')
            ge_mid = ((ge_p1[0] + ge_p2[0]) / 2, (ge_p1[1] + ge_p2[1]) / 2)
            for mia, (mp1, mp2, _, _) in enumerate(merged_edges):
                mmid = ((mp1[0] + mp2[0]) / 2, (mp1[1] + mp2[1]) / 2)
                d = (ge_mid[0] - mmid[0])**2 + (ge_mid[1] - mmid[1])**2
                if d < best_mid_dist:
                    best_mid_dist = d
                    best_mia = mia

            candidate_merged = merge_polygons(merged_poly, poly_aligned, best_mia, ib)

            # ---- 面积校验 ----
            merged_area = float(cv2.contourArea(candidate_merged.astype(np.float32)))
            frag_areas = sum(float(cv2.contourArea(f.astype(np.float32)))
                             for _, f in display_frags)
            frag_areas += float(cv2.contourArea(poly_aligned.astype(np.float32)))

            area_ratio = (min(merged_area, frag_areas) /
                          max(merged_area, frag_areas) if max(merged_area, frag_areas) > 0 else 0.0)

            print(f"    融合面积={merged_area:.0f}  碎片面积和={frag_areas:.0f}  "
                  f"比值={area_ratio:.3f}  (阈值={area_threshold})")

            if area_ratio >= area_threshold:
                print(f"    ✓ 通过")
                display_frags.append((orig_idx, poly_aligned))
                merged_poly = candidate_merged
                accepted = True
                break
            else:
                print(f"    ✗ 面积差过大，加入黑名单")
                blacklist.add(pair_key)

        if not accepted:
            print(f"  ⚠ 碎片{orig_idx+1} 所有 {len(scored_pairs)} 组边对均未通过校验，跳过")

    print(f"\n最终结果: {len(merged_poly)} 顶点, {len(display_frags)} 碎片")

    # ---- 3. 绘制 ----
    canvas = draw_result(display_frags, merged_poly)
    return canvas, [merged_poly]


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


def draw_result(fragments, merged=None):
    """
    绘制拼接示意图。

    - 彩色半透明填充 + 轮廓 = 各碎片（对齐后的位置）
    - 白色粗轮廓 = 融合后的最终多边形（如果提供）
    - #1 #2 ... 标注选择顺序
    """
    all_pts_list = [f for _, f in fragments]
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
    for i, (_, frag) in enumerate(fragments):
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

        canvas, _ = reassemble(polygons)

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
