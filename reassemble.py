#!/usr/bin/env python3
"""
碎片拼接 — 多碎片版

逻辑：
  1. 从图像中检测 N 个多边形碎片
  2. 固定面积最大的那个
  3. 逐轮：枚举"已融合多边形"与"剩余碎片"之间的所有边对，选长度最接近的一组
  4. 中点对齐（共享垂直平分线）→ 顶点融合 → 更新融合结果
  5. 重复直到所有碎片合并完毕
  6. 在窗口显示拼接结果
"""

import numpy as np
import cv2
import sys
import os


# ============================================================
#  几何工具
# ============================================================

def extract_hull(mask: np.ndarray, approx_epsilon: float = 8.0) -> np.ndarray | None:
    """从二值 mask 提取简化凸包，返回 (N, 2) 顶点数组。"""
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, approx_epsilon, True)
    hull = cv2.convexHull(approx)
    return hull.reshape(-1, 2)


def hull_edges(hull: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """返回 [(p1, p2, length), ...]，每个元素是一条边。"""
    edges = []
    n = len(hull)
    for i in range(n):
        p1 = hull[i].astype(np.float64)
        p2 = hull[(i + 1) % n].astype(np.float64)
        length = float(np.linalg.norm(p2 - p1))
        edges.append((p1, p2, length))
    return edges


def merge_collinear_edges(hull: np.ndarray, angle_tol: float = 3.0) -> np.ndarray:
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
    if len(kept) < 3:
        return hull
    return kept


# ============================================================
#  边匹配 & 仿射对齐
# ============================================================

def find_closest_edge_pair(edges_a: list, edges_b: list) -> tuple[int, int]:
    """返回 (idx_a, idx_b) — 长度最接近的一对边。"""
    best_diff = float("inf")
    best_pair = (0, 0)

    for ia, (_, _, la) in enumerate(edges_a):
        for ib, (_, _, lb) in enumerate(edges_b):
            diff = abs(la - lb)
            if diff < best_diff:
                best_diff = diff
                best_pair = (ia, ib)

    return best_pair


def find_top_edge_pairs(edges_a: list, edges_b: list,
                        k: int | None = None) -> list[tuple[int, int, float]]:
    """返回长度差最小的 k 组边对 [(idx_a, idx_b, diff), ...]，按 diff 升序排列。
    若 k=None，返回所有 M×N 对。"""
    pairs = []
    for ia, (_, _, la) in enumerate(edges_a):
        for ib, (_, _, lb) in enumerate(edges_b):
            diff = abs(la - lb)
            pairs.append((ia, ib, diff))
    pairs.sort(key=lambda x: x[2])
    if k is None:
        return pairs
    return pairs[:k]


def align_matrix(edge_from: tuple, edge_to: tuple) -> np.ndarray:
    """
    计算 2x3 仿射矩阵，把 edge_from 对齐到 edge_to。

    对齐方式：两边中点重合、方向相反、共享同一条垂直平分线。
    短边居中落在长边上，不要求端点重合。
    """
    p1_f, p2_f, _ = edge_from
    p1_t, p2_t, _ = edge_to

    mf = (p1_f + p2_f) / 2.0
    mt = (p1_t + p2_t) / 2.0

    vf = p2_f - p1_f
    vt = p2_t - p1_t

    nf = vf / np.linalg.norm(vf)
    nt = vt / np.linalg.norm(vt)

    # 旋转 nf 到 -nt（切割边方向相反）
    target = -nt
    cos_t = np.dot(nf, target)
    sin_t = np.cross(nf, target)

    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]])
    t = mt - R @ mf

    return np.column_stack([R, t])


def transform(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """对 (N, 2) 点集应用 2x3 仿射变换。"""
    R, t = M[:, :2], M[:, 2]
    return (R @ pts.T).T + t


# ============================================================
#  顶点融合
# ============================================================

def merge_polygons(fixed: np.ndarray,
                   moving_aligned: np.ndarray,
                   ia_fixed: int,
                   ib_moving: int) -> np.ndarray:
    """
    融合两个对齐后的多边形。

    端点交叉配对（方向相反）→ 取中点 → 排除原始贴合边端点 →
    加融合顶点 → 凸包 → 简化 → 去共线。
    """
    n_f, n_m = len(fixed), len(moving_aligned)

    idx_f1 = ia_fixed
    idx_f2 = (ia_fixed + 1) % n_f
    idx_m1 = ib_moving
    idx_m2 = (ib_moving + 1) % n_m

    # 交叉配对取中点
    v1 = (fixed[idx_f1] + moving_aligned[idx_m2]) / 2.0   # 固定首 ↔ 移动尾
    v2 = (fixed[idx_f2] + moving_aligned[idx_m1]) / 2.0   # 固定尾 ↔ 移动首

    pts = [v1, v2]
    for i in range(n_f):
        if i != idx_f1 and i != idx_f2:
            pts.append(fixed[i])
    for i in range(n_m):
        if i != idx_m1 and i != idx_m2:
            pts.append(moving_aligned[i])

    all_pts = np.array(pts, dtype=np.float32)

    merged = cv2.convexHull(all_pts).reshape(-1, 2)
    peri = cv2.arcLength(merged.astype(np.float32), True)
    merged = cv2.approxPolyDP(merged.astype(np.float32), 0.02 * peri, True)
    merged = merged.reshape(-1, 2)
    merged = merge_collinear_edges(merged)

    return merged


# ============================================================
#  可视化
# ============================================================

# 碎片着色（按面积从大到小）
FRAGMENT_COLORS = [
    (180, 120, 60),   # 蓝
    (60, 160, 210),   # 橙
    (80, 200, 120),   # 绿
    (120, 80, 200),   # 紫
    (200, 160, 80),   # 青
]


def draw_result(fragments: list[np.ndarray],
                merged: np.ndarray) -> np.ndarray:
    """绘制拼接示意图：彩色半透明=各碎片, 白色粗线=融合轮廓。"""
    all_pts = np.vstack(fragments + [merged]).astype(np.int32)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    margin = 60
    w = x_max - x_min + 2 * margin
    h = y_max - y_min + 2 * margin
    offset = np.array([-x_min + margin, -y_min + margin])

    canvas = np.full((h, w, 3), 30, dtype=np.uint8)

    # ---- 各碎片 (半透明) ----
    for i, frag in enumerate(fragments):
        color = FRAGMENT_COLORS[i % len(FRAGMENT_COLORS)]
        frag_s = (frag + offset).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [frag_s], color)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
        cv2.polylines(canvas, [frag_s], True, color, 2)

    # ---- 融合后多边形 (白色粗轮廓) ----
    merged_s = (merged + offset).astype(np.int32)
    cv2.polylines(canvas, [merged_s], True, (255, 255, 255), 3)

    return canvas


# ============================================================
#  公开 API
# ============================================================

def masks_from_yolo(results, num_fragments: int = 4) -> list[np.ndarray]:
    """从 YOLOv8 推理结果中提取 mask 列表。"""
    r = results[0]
    if r.masks is None:
        raise ValueError("未检测到任何目标")

    h, w = r.orig_shape
    masks = []

    for mask_tensor in r.masks.data[:num_fragments]:
        m = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append(m)

    return masks


def reassemble(masks: list[np.ndarray]) -> tuple[np.ndarray, list]:
    """
    拼接 N 个碎片：按面积排序 → 固定最大 → 逐轮匹配+融合。

    Returns:
        (canvas, [merged_poly])
    """
    # ---- 1. 提取所有凸包 ----
    hulls = []
    for i, m in enumerate(masks):
        hull = extract_hull(m)
        if hull is None:
            raise ValueError(f"碎片 {i} 无法提取轮廓")
        hulls.append(hull)

    # ---- 2. 按面积升序排列（最小固定） ----
    areas = [cv2.contourArea(h.astype(np.float32)) for h in hulls]
    order = sorted(range(len(hulls)), key=lambda i: areas[i])
    sorted_hulls = [hulls[i] for i in order]

    for idx, o in enumerate(order):
        print(f"碎片 {o}: {len(hulls[o])} 顶点, 面积 ≈ {areas[o]:.0f}")

    # ---- 3. 合并核心逻辑（内联，打印过程） ----
    def do_merge(hulls_subset: list) -> tuple[np.ndarray, list]:
        """合并 hull 子集，返回 (merged_poly, display_polys)。"""
        display = [hulls_subset[0]]
        merged_poly = hulls_subset[0]

        for i in range(1, len(hulls_subset)):
            moving = hulls_subset[i]
            merged_edges = hull_edges(merged_poly)
            moving_edges = hull_edges(moving)

            ia, ib = find_closest_edge_pair(merged_edges, moving_edges)
            la = merged_edges[ia][2]
            lb = moving_edges[ib][2]
            diff = abs(la - lb)
            print(f"\n第 {i} 次融合: merged[{ia}]={la:.1f}px  <->  "
                  f"fragment[{ib}]={lb:.1f}px  (差 {diff:.1f}px)")

            M = align_matrix(moving_edges[ib], merged_edges[ia])
            moving_aligned = transform(M, moving)
            display.append(moving_aligned)
            merged_poly = merge_polygons(merged_poly, moving_aligned, ia, ib)
            print(f"  → 融合后 {len(merged_poly)} 顶点")

        return merged_poly, display

    merged, display = do_merge(sorted_hulls)
    print(f"\n最终结果: {len(merged)} 顶点")

    # ---- 4. 矩形检测 & 回退搜索 ----
    is_rect = is_rectangle_by_iou(merged, verbose=False)

    if not is_rect:
        print("\n⚠ 全量拼接结果不是矩形，启动回溯搜索...")

        # 4a. 先尝试全量碎片 + 回溯所有边对
        result = reassemble_search(sorted_hulls)
        if result is not None:
            return result

        # 4b. 尝试丢弃最大碎片（最后 1~2 块）
        for drop in range(1, min(3, len(sorted_hulls) - 1)):
            subset = sorted_hulls[:len(sorted_hulls) - drop]
            print(f"\n--- 尝试去掉最后 {drop} 块碎片（共 {len(subset)} 块） ---")
            result = reassemble_search(subset)
            if result is not None:
                return result

        print("\n⚠ 未找到矩形组合，返回原始全量拼接结果")

    # ---- 5. 渲染 ----
    canvas = draw_result(display, merged)
    return canvas, [merged]


def reassemble_search(sorted_hulls: list[np.ndarray],
                      pair_candidates: int | None = None) -> tuple[np.ndarray, list] | None:
    """
    DFS 回溯搜索：尝试不同的边配对组合，直到拼出矩形。

    优先回溯最后加入的碎片（最大碎片），逐级向前尝试备选边对。
    默认尝试所有 M×N 边对（k=None），保证穷举搜索。

    Args:
        sorted_hulls:    按面积升序排列的 hull 列表
        pair_candidates: 每次合并尝试的候选边对数量，None=全部

    Returns:
        (canvas, [merged_poly]) 如果找到矩形，否则 None
    """
    n = len(sorted_hulls)
    tried_count = [0]  # mutable counter

    def dfs(idx: int, merged_poly: np.ndarray, display_polys: list,
            history: list) -> tuple[np.ndarray, list, list] | None:
        """
        history: [(step, ia, ib, diff, rank), ...]
        从 idx 开始继续合并，返回 (merged_poly, display_polys, history) 或 None
        """
        if idx >= n:
            # 所有碎片已合并 → 静默检测矩形
            if is_rectangle_by_iou(merged_poly, verbose=False):
                return merged_poly, display_polys, history
            return None

        moving = sorted_hulls[idx]
        merged_edges = hull_edges(merged_poly)
        moving_edges = hull_edges(moving)

        pairs = find_top_edge_pairs(merged_edges, moving_edges, k=pair_candidates)
        total_pairs = len(pairs)

        for rank, (ia, ib, diff) in enumerate(pairs):
            if rank > 0:
                tried_count[0] += 1
                print(f"  [回溯] 第 {idx} 次融合尝试备选边对 #{rank}/{total_pairs-1}: "
                      f"merged[{ia}] <-> fragment[{ib}] (差 {diff:.1f}px)")

            M = align_matrix(moving_edges[ib], merged_edges[ia])
            moving_aligned = transform(M, moving)
            new_display = display_polys + [moving_aligned]
            new_merged = merge_polygons(merged_poly, moving_aligned, ia, ib)
            new_history = history + [(idx, ia, ib, diff, rank)]

            result = dfs(idx + 1, new_merged, new_display, new_history)
            if result is not None:
                return result

        return None

    # ---- 启动 DFS ----
    print(f"  [搜索] 共 {n} 块碎片，{n-1} 次合并，穷举搜索中...")
    result = dfs(1, sorted_hulls[0], [sorted_hulls[0]], [])

    if result is not None:
        merged_poly, display_polys, history = result
        print(f"\n[搜索成功] 共尝试 {tried_count[0]} 种备选方案，"
              f"找到可拼成矩形的边配对组合:")
        for step, ia, ib, diff, rank in history:
            marker = " (备选)" if rank > 0 else ""
            print(f"  第 {step} 次融合: merged[{ia}] <-> fragment[{ib}] "
                  f"差={diff:.1f}px{marker}")
        canvas = draw_result(display_polys, merged_poly)
        return canvas, [merged_poly]

    print(f"[搜索失败] 尝试了 {tried_count[0]} 种备选方案，未找到矩形组合")
    return None


# ============================================================
#  面积 IoU 矩形检测
# ============================================================

def is_rectangle_by_iou(polygon: np.ndarray, threshold: float = 0.9,
                        verbose: bool = True) -> bool:
    """
    面积 IoU 法判断多边形是否接近矩形。

    原理：多边形面积 / 最小外接旋转矩形面积。
    比值越接近 1.0，形状越接近矩形。

    Args:
        polygon:  (N, 2) 顶点数组
        threshold: IoU 阈值，超过此值判定为矩形
        verbose:  是否打印详细信息（DFS 搜索时关闭以减少噪音）

    Returns:
        True 如果形状接近矩形
    """
    if len(polygon) < 4:
        return False

    pts = polygon.astype(np.float32)

    # 多边形面积
    poly_area = float(cv2.contourArea(pts))

    # 最小外接旋转矩形
    rect = cv2.minAreaRect(pts)
    rect_area = float(rect[1][0] * rect[1][1])  # width * height

    if rect_area < 1.0:
        return False

    iou = poly_area / rect_area

    if verbose:
        print(f"[矩形检测] 多边形面积 = {poly_area:.0f} px², "
              f"外接矩形面积 = {rect_area:.0f} px², "
              f"IoU = {iou:.3f}")

    if iou >= threshold:
        if verbose:
            center = rect[0]
            size = rect[1]
            angle = rect[2]
            print(f"  ✓ 检测结果为矩形! "
                  f"(中心=({center[0]:.1f}, {center[1]:.1f}), "
                  f"尺寸=({size[0]:.1f}, {size[1]:.1f}), "
                  f"角度={angle:.1f}°)")
        return True
    else:
        if verbose:
            print(f"  ✗ 不是矩形 (IoU < {threshold})")
        return False


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
        masks = masks_from_yolo(results)
        canvas, merged_polys = reassemble(masks)

        # ---- 面积 IoU 矩形检测 ----
        if merged_polys and len(merged_polys) > 0:
            is_rectangle_by_iou(merged_polys[0])

        cv2.imshow("Reassembly", canvas)
        cv2.waitKey(0)
        cv2.imwrite("reassembled.jpg", canvas)
        print("已保存: reassembled.jpg")
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
