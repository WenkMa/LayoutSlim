import numpy as np
import cv2

# 设置图像路径
image_path = "file3.jpg"
image = cv2.imread(image_path)
# 图像高度和宽度（仅用于覆盖数组初始化，实际会自动读取图像尺寸）
IMAGE_HEIGHT, IMAGE_WIDTH =image.shape[:2]
# ----------- 输入：检测框 Y 范围和 X 范围 -----------
# 所有 Y 方向的检测框 (y1, y2)
import re

# 原始 BBox 文本
bbox_text = """
BBox: (139, 159, 805, 605), Class: 1, Confidence: 0.9833
BBox: (139, 1014, 807, 1321), Class: 1, Confidence: 0.9831
BBox: (849, 885, 1515, 1193), Class: 1, Confidence: 0.9820
BBox: (139, 1325, 808, 1563), Class: 1, Confidence: 0.9801
BBox: (850, 160, 1516, 673), Class: 1, Confidence: 0.9796
BBox: (139, 1716, 807, 1990), Class: 1, Confidence: 0.9790
BBox: (849, 677, 1515, 881), Class: 1, Confidence: 0.9785
BBox: (849, 1232, 1515, 1781), Class: 1, Confidence: 0.9762
BBox: (849, 1786, 1514, 1989), Class: 1, Confidence: 0.9710
BBox: (139, 646, 806, 1009), Class: 1, Confidence: 0.9461
BBox: (139, 91, 462, 116), Class: 2, Confidence: 0.9201
BBox: (140, 1642, 365, 1678), Class: 0, Confidence: 0.9102
BBox: (1396, 2042, 1512, 2076), Class: 2, Confidence: 0.8967
BBox: (140, 632, 806, 663), Class: 0, Confidence: 0.3681
"""

# 使用正则表达式提取 (x1, y1, x2, y2)
pattern = re.compile(r"BBox:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)")
matches = pattern.findall(bbox_text)

# 提取 bbox_y 和 bbox_x
bbox_y = []
bbox_x = []

for match in matches:
    x1, y1, x2, y2 = map(int, match)
    bbox_y.append((y1, y2))
    bbox_x.append((x1, x2))
# ---------- 工具函数：合并连续未覆盖区间 ----------
def merge_ranges(coords):
    if len(coords) == 0:
        return []
    ranges = []
    start = prev = coords[0]
    for x in coords[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))
    return ranges

# ---------- 找出 Y 方向未覆盖的区域 ----------
def get_uncovered_y_ranges(height, bboxes):
    coverage = np.zeros(height, dtype=bool)
    for y1, y2 in bboxes:
        y1 = max(0, y1)
        y2 = min(height - 1, y2)
        coverage[y1:y2+1] = True
    uncovered_rows = np.where(coverage == False)[0]
    return merge_ranges(uncovered_rows)

# ---------- 找出 X 方向未覆盖的区域 ----------
def get_uncovered_x_ranges(width, bboxes):
    coverage = np.zeros(width, dtype=bool)
    for x1, x2 in bboxes:
        x1 = max(0, x1)
        x2 = min(width - 1, x2)
        coverage[x1:x2+1] = True
    uncovered_cols = np.where(coverage == False)[0]
    return merge_ranges(uncovered_cols)

# ---------- 裁剪图像的函数 ----------
def crop_blank_regions(image, blank_y, blank_x):
    # 裁剪纵向（Y轴）
    cut_y = 0
    for y1, y2 in blank_y:
        y1_cut = max(0, y1 - cut_y)
        y2_cut = max(0, y2 - cut_y)
        image = np.concatenate((image[:y1_cut, :], image[y2_cut+1:, :]), axis=0)
        cut_y += (y2 - y1 + 1)

    # 裁剪横向（X轴）
    cut_x = 0
    for x1, x2 in blank_x:
        x1_cut = max(0, x1 - cut_x)
        x2_cut = max(0, x2 - cut_x)
        image = np.concatenate((image[:, :x1_cut], image[:, x2_cut+1:]), axis=1)
        cut_x += (x2 - x1 + 1)

    return image

# ---------- 主流程 ----------
if __name__ == "__main__":
    # 读取图像

    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    h, w = image.shape[:2]

    # 获取未覆盖的区域
    blank_y_ranges = get_uncovered_y_ranges(h, bbox_y)
    blank_x_ranges = get_uncovered_x_ranges(w, bbox_x)

    print("未覆盖 Y 区间：", blank_y_ranges)
    print("未覆盖 X 区间：", blank_x_ranges)

    # 裁剪图像
    cropped = crop_blank_regions(image, blank_y_ranges, blank_x_ranges)

    # 保存结果
    cv2.imwrite("out_xy.jpg", cropped)
    print("已保存裁剪后的图像为 out_xy.jpg")
