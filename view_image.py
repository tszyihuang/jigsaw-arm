#!/usr/bin/python3
"""独立进程图片查看器 — 单次弹出窗口, 关掉不影响调用方进程.

用法: python3 view_image.py <image_path> [window_title]

按 q / Esc 或点击窗口右上角 X 关闭.
由其他程序 (如 infer.py 的 read 流程) 通过 subprocess.Popen 以独立进程
方式调用, 窗口的打开/关闭/崩溃都只在子进程内, 不影响主进程.
"""

import sys

import cv2


def main():
    if len(sys.argv) < 2:
        print("用法: python3 view_image.py <image_path> [window_title]")
        sys.exit(1)
    path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else path

    img = cv2.imread(path)
    if img is None:
        print(f"无法读取图像: {path}")
        sys.exit(1)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, img.shape[1], img.shape[0])
    cv2.imshow(title, img)
    print(f"窗口 '{title}' 已弹出 — 按 q / Esc 或点击右上角 X 关闭")
    seen_visible = False
    try:
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord('q'), 27):
                break
            try:
                gone = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
            except cv2.error:
                gone = True
            if gone:
                # 仅当窗口"曾经可见"后才视为用户关闭, 避免初始化竞态
                # (窗口刚创建尚未显示时 visible 可能短暂为 0) 导致一闪而过
                if seen_visible:
                    break
            else:
                seen_visible = True
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
