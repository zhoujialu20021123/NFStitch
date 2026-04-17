import cv2
import numpy as np
import os
import subprocess
from skimage.segmentation import flood_fill
from skimage.measure import label, regionprops

# 等价于 FindStEnPintsAndRegion 函数
def find_start_end_points_and_region(img1, img2):
    """
    在两张农田图像中找到重叠区域、起点和终点
    特别针对土壤裂纹、植物幼苗和灌溉管道优化
    """
    # 创建掩码
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mask1 = np.uint8((gray1 > 0).astype(np.uint8))
    mask2 = np.uint8((gray2 > 0).astype(np.uint8))
    
    # 找出重叠区域
    overlap_mask = mask1 & mask2
    
    # 找出左右边界 (处理农田边界)
    col_max = np.max(overlap_mask, axis=0)
    left = np.argmax(col_max) 
    # 反向搜索右边界 - 从右侧开始找第一个非零值
    right = len(col_max) - np.argmax(col_max[::-1] > 0) - 1
    
    # 找出上下边界
    row_max = np.max(overlap_mask, axis=1)
    top = np.argmax(row_max)
    # 反向搜索下边界
    bottom = len(row_max) - np.argmax(row_max[::-1] > 0) - 1
    
    # 确保有有效的重叠区域
    if left >= right or top >= bottom:
        print("错误：找不到有效的重叠区域！")
        return None
    
    # 提取重叠区域
    region_img1 = img1[top:bottom, left:right]
    region_img2 = img2[top:bottom, left:right]
    region_mask = overlap_mask[top:bottom, left:right]
    
    # 检测管道特征作为起点
    # 检测图像1中的黑色管道起点
    black_lower = np.array([0, 0, 0], dtype="uint8")
    black_upper = np.array([50, 50, 50], dtype="uint8")
    pipeline_mask1 = cv2.inRange(region_img1, black_lower, black_upper)
    # 检测图像2中的黑色管道起点
    pipeline_mask2 = cv2.inRange(region_img2, black_lower, black_upper)
    
    # 创建起点和终点标记 (三维矩阵)
    points = np.zeros((*region_mask.shape, 2), dtype=np.uint8)
    
    # 如果在左图中检测到管道起点
    if np.any(pipeline_mask1):
        start_y, start_x = np.unravel_index(np.argmax(pipeline_mask1), pipeline_mask1.shape)
        points[start_y, start_x, 0] = 1
        print(f"使用管道起点坐标: ({start_x}, {start_y})")
    else:
        # 默认起点
        points[10, 10, 0] = 1
        print("使用默认起点")
    
    # 终点检测（使用土壤边缘检测）
    soil_edges = cv2.Canny(cv2.cvtColor(region_img2, cv2.COLOR_BGR2GRAY), 50, 150)
    
    # 强化土壤裂缝特征
    kernel = np.ones((5, 5), np.uint8)
    soil_edges = cv2.dilate(soil_edges, kernel, iterations=1)
    
    if np.any(soil_edges):
        # 寻找最右侧的边沿点
        soil_points = np.where(soil_edges > 0)
        if len(soil_points[0]) > 0 and len(soil_points[1]) > 0:
            max_idx = np.argmax(soil_points[1])
            end_y, end_x = soil_points[0][max_idx], soil_points[1][max_idx]
            points[end_y, end_x, 1] = 1
            print(f"使用土壤边缘终点: ({end_x}, {end_y})")
    
    # 线性索引
    start_idx = np.argwhere(points[:, :, 0] > 0)[0]
    end_idx = np.argwhere(points[:, :, 1] > 0)[0]
    twop = (start_idx, end_idx)
    
    return points, region_img1, region_img2, region_mask, left, top, twop

# 等价于 SeamEstimation 函数
def estimate_seam(im1, im2, points, twop):
    """
    在农田重叠区域中估计最佳拼接缝
    特别处理土壤裂纹、植物幼苗和管道特征
    """
    # 1. 颜色差异计算 (强化土壤裂缝的颜色差异)
    color_diff = cv2.absdiff(im1.astype(np.float32)/255, im2.astype(np.float32)/255)
    
    # 针对农田图像 - 强调土壤区域的颜色变化
    color_diff_gray = np.mean(color_diff, axis=2)
    
    # 2. 结构差异计算 (使用Sobel边缘检测)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    # 植物幼苗保护 - 增强对比度
    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)
    
    sobel_x1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_mag1 = np.sqrt(sobel_x1**2 + sobel_y1**2)
    grad_mag2 = np.sqrt(sobel_x2**2 + sobel_y2**2)
    
    struct_diff = np.abs(grad_mag1 - grad_mag2)
    struct_diff = cv2.normalize(struct_diff, None, 0, 1, cv2.NORM_MINMAX)
    
    # 3. 线检测 (使用LSD算法检测管道)
    lsd = cv2.createLineSegmentDetector()
    lines1, _, _, _ = lsd.detect(gray1)
    lines2, _, _, _ = lsd.detect(gray2)
    
    # 创建线掩码
    line_mask = np.zeros_like(gray1, dtype=np.float32)
    if lines1 is not None:
        for line in lines1:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, 2)
    if lines2 is not None:
        for line in lines2:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, 2)
    
    # 4. 组合能量函数 (针对农田优化权重)
    # 降低管道区域的权重 - 避免切割管道
    line_mask[line_mask > 0] = 0.3  # 降低权重
    
    # 增加植物区域的权重 - 保护绿色幼苗
    green_lower = np.array([30, 60, 20], dtype=np.uint8)
    green_upper = np.array([90, 200, 120], dtype=np.uint8)
    plant_mask = cv2.inRange(cv2.cvtColor(im1, cv2.COLOR_BGR2HSV), 
                             green_lower, green_upper)
    plant_mask = plant_mask.astype(np.float32) / 255
    plant_mask[plant_mask > 0] = 2.0  # 增加权重
    
    # 组合能量函数
    energy = color_diff_gray * 0.4 + struct_diff * 0.3 + line_mask * 0.2 + plant_mask * 0.1
    energy = cv2.normalize(energy, None, 0, 1, cv2.NORM_MINMAX)
    
    # 5. 动态规划查找路径 (保护植物幼苗)
    start_y, start_x = np.argwhere(points[:, :, 0] > 0)[0]
    end_y, end_x = np.argwhere(points[:, :, 1] > 0)[0]
    
    # 创建路径成本矩阵
    path_matrix = np.full(energy.shape, np.inf)
    path_matrix[start_y, start_x] = energy[start_y, start_x]
    
    # 定义可移动方向
    dx = [0, 1, 1, 1]
    dy = [1, 0, 1, -1]
    
    # 创建路径网格
    height, width = energy.shape
    q = [(start_y, start_x)]
    
    while q:
        y, x = q.pop(0)
        
        # 到达终点
        if y == end_y and x == end_x:
            break
            
        for i in range(4):
            ny, nx = y + dy[i], x + dx[i]
            
            if 0 <= ny < height and 0 <= nx < width:
                cost = path_matrix[y, x] + energy[ny, nx]
                
                # 如果发现更短的路径
                if cost < path_matrix[ny, nx]:
                    path_matrix[ny, nx] = cost
                    q.append((ny, nx))
    
    # 回溯构建路径
    path = []
    y, x = end_y, end_x
    seam_mask = np.zeros_like(energy)
    
    while (y, x) != (start_y, start_x):
        path.append((y, x))
        seam_mask[y, x] = 1
        
        # 找到最小成本的邻点
        min_cost = np.inf
        next_point = (y, x)
        for i in range(4):
            py, px = y - dy[i], x - dx[i]
            if 0 <= py < height and 0 <= px < width:
                if path_matrix[py, px] < min_cost:
                    min_cost = path_matrix[py, px]
                    next_point = (py, px)
        
        y, x = next_point
    
    path.append((start_y, start_x))
    seam_mask[start_y, start_x] = 1
    
    return seam_mask

def seam_cut_and_blend(img1, img2, seam_mask, dx, dy):
    """
    执行拼接缝裁剪和融合
    """
    height, width = img1.shape[:2]
    
    # 创建裁剪后的拼接缝掩码
    seam_full = np.zeros((height, width), dtype=np.uint8)
    seam_full[dy:dy+seam_mask.shape[0], dx:dx+seam_mask.shape[1]] = seam_mask
    
    # 创建图像掩码
    mask1 = (seam_full == 0).astype(np.uint8)
    mask2 = (seam_full == 1).astype(np.uint8)
    
    # 连通区域处理 - 优化土壤区域的连续性
    labels1 = label(mask1)
    largest_cc1 = max(regionprops(labels1), key=lambda x: x.area).label
    mask1[labels1 != largest_cc1] = 0
    
    labels2 = label(mask2)
    largest_cc2 = max(regionprops(labels2), key=lambda x: x.area).label
    mask2[labels2 != largest_cc2] = 0
    
    # 保存带透明通道的图像
    img1_with_alpha = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
    img1_with_alpha[:, :, 3] = mask1 * 255
    
    img2_with_alpha = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
    img2_with_alpha[:, :, 3] = mask2 * 255
    
    cv2.imwrite("img1_with_alpha.png", img1_with_alpha)
    cv2.imwrite("img2_with_alpha.png", img2_with_alpha)
    
    # 使用enblend进行融合 (确保安装了enblend)
    cmd = [
        "enblend", 
        "-o", "blended_result.png", 
        "--gpu",
        "img1_with_alpha.png", "img2_with_alpha.png"
    ]
    subprocess.run(cmd, check=True)
    
    # 加载融合结果
    result = cv2.imread("blended_result.png", cv2.IMREAD_UNCHANGED)
    
    # 添加拼接缝标记用于可视化
    if result is not None and len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    if result is not None:
        result_with_seam = result.copy()
        seam_mask = seam_mask.astype(np.uint8) * 255
        result_with_seam[dy:dy+seam_mask.shape[0], dx:dx+seam_mask.shape[1], 0] = \
            np.maximum(result_with_seam[dy:dy+seam_mask.shape[0], dx:dx+seam_mask.shape[1], 0], seam_mask)
    
    return result, result_with_seam

# 主处理流程
def main():
    # 读取农田图像
    img1_path = "/data/zhou/Result/Wheat_Test/Result/result_000.png"
    img2_path = "/data/zhou/Result/Wheat_Test/Result/result_001.png"
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("错误：无法加载图像！")
        return
    
    # 寻找起点、终点和重叠区域
    result = find_start_end_points_and_region(img1, img2)
    
    if result is None:
        print("错误：无法找到重叠区域！")
        return
    
    points, region_img1, region_img2, region_mask, dx, dy, twop = result
    
    # 估计拼接缝
    seam_mask = estimate_seam(region_img1, region_img2, points, twop)
    
    # 执行裁剪和融合
    blended_result, result_with_seam = seam_cut_and_blend(img1, img2, seam_mask, dx, dy)
    
    # 保存结果
    if blended_result is not None:
        cv2.imwrite("final_blend.png", blended_result)
        print("拼接结果已保存为 final_blend.png")
    
    if result_with_seam is not None:
        cv2.imwrite("blend_with_seam.png", result_with_seam)
        print("带拼接缝的结果已保存为 blend_with_seam.png")

if __name__ == "__main__":
    main()