import numpy as np
import os
import glob
import cv2
import sys
import math
import RANSAC
import random
import colorsys
from enum import Enum
from scipy.optimize import lsq_linear
# from cv2.xfeatures2d import matchGMS
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageChops, ImageOps
# from osgeo import osr, gdal


class Transformation(Enum):
    translation = 1
    similarity = 2
    affine = 3
    homography = 4
    full = 5
def Add_WaterMark(image):
    
    args = {
        'file': 'Water/results.png', # image file path or directory"
        "mark": "西北农林科技大学", # watermark content
        'out': './output', # image output directory

        'color': '#00FF00',
        'space': 150,
        'angle': 30,

        'font-family':'font.ttf',
        'font-height-crop':'1.2',
        'size':50,
        'opacity':0.5,
        'quality': 100,
    }
    if isinstance(args["mark"], str) and sys.version_info[0] < 3:
        args["mark"] = args["mark"].decode("utf-8")

    mark = gen_mark(args)

    result = add_mark(image, mark, args)

    return result

def add_mark(im, mark, args):
    '''
    添加水印，然后保存图片
    '''
    # im = Image.open(imagePath)
    im = ImageOps.exif_transpose(im)

    image = mark(im)
    name = 'ortho-mask.png'
    if image:
        
        new_name = os.path.join(args["out"], name)
        if os.path.splitext(new_name)[1] != '.png':
            image = image.convert('RGB')
        print(name + " Success.")
        return image
    else:
        print(name + " Failed.")


def set_opacity(im, opacity):
    '''
    设置水印透明度
    '''
    assert opacity >= 0 and opacity <= 1

    alpha = im.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    im.putalpha(alpha)
    return im


def crop_image(im):
    '''裁剪图片边缘空白'''
    bg = Image.new(mode='RGBA', size=im.size)
    diff = ImageChops.difference(im, bg)
    del bg
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im

def compute_overlap(H1, H2, w1, h1, w2, h2):
    """
    计算两个图像之间的重叠率（交集面积占图像1面积的比例）
    
    参数:
        H1: 图像1的变换矩阵（3x3）
        H2: 图像2的变换矩阵（3x3）
        w1: 图像1的宽度
        h1: 图像1的高度
        w2: 图像2的宽度
        h2: 图像2的高度
    
    返回:
        overlap_ratio: 重叠率（0到1之间）
    """
    # 定义图像1的角点（顺时针顺序）
    corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32)
    # 将角点转换为齐次坐标并应用变换矩阵H1
    pts1 = cv2.perspectiveTransform(corners1.reshape(-1, 1, 2), H1).squeeze()
    
    # 定义图像2的角点
    corners2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32)
    pts2 = cv2.perspectiveTransform(corners2.reshape(-1, 1, 2), H2).squeeze()
    
    # 计算两个凸多边形的交集面积
    area_intersection, _ = cv2.intersectConvexConvex(pts1, pts2)
    area_intersection = abs(area_intersection)  # 确保面积为正
    
    # 计算图像1的面积
    area1 = w1 * h1
    
    # 计算重叠率
    overlap_ratio = area_intersection / area1 if area1 > 0 else 0
    
    return overlap_ratio


def grid_based_point_selection(matches, inliers, H, img2_shape, kp_1, kp_2, grid_size=(10, 10), P=50, scale=1, numa = 1, numb = 1, X_run = 1, Y_run = 1):
    """
    在重叠区域划分网格并随机选择P对匹配点
    
    参数:
        matches: 匹配点列表 (DMatch对象)
        H: 单应性矩阵 (3x3)
        img2_shape: 图像2的尺寸 (height, width)
        grid_size: 网格划分尺寸 (rows, cols)
        P: 需要选择的匹配点对数
        
    返回:
        selected_matches: 选中的P对匹配点
    """
    # 1. 计算图像2在图像1坐标系中的投影区域
    # h1, w1 = img1_shape[:2]
    h2, w2 = img2_shape[:2]
    h1 = h2
    w1 = w2
    # print(h1, w1)
    
    corners_img2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32)
    warped_corners = cv2.perspectiveTransform(corners_img2.reshape(-1, 1, 2), H)

    # # 2. 计算精确重叠面积（使用多边形交集）
    # # 定义图像1的多边形（凸多边形）
    # poly_img1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32)
    # # 定义变换后图像2的多边形（确保凸性，必要时使用凸包）
    # poly_img2 = warped_corners
    # # 计算交集面积
    # intersection_area, _ = cv2.intersectConvexConvex(poly_img1, poly_img2)
    # intersection_area = abs(intersection_area)  # 面积取正
    
    # # 计算各区域面积
    # area_img1 = w1 * h1
    # area_img2 = w2 * h2
    # union_area = area_img1 + area_img2 - intersection_area  # 并集面积
    
    # # 计算多种重叠率指标
    # overlap_ratio_img1 = intersection_area / area_img1  # 交集占图像1的比例
    # overlap_ratio_img2 = intersection_area / area_img2  # 交集占图像2的比例
    # iou = intersection_area / union_area  # 交并比（IoU）
    
    # print(f"精确交集面积: {intersection_area:.2f} pixels")
    # print(f"重叠率（相对于图像1）: {overlap_ratio_img1 * 100:.2f}%")
    # print(f"重叠率（相对于图像2）: {overlap_ratio_img2 * 100:.2f}%")
    # print(f"IoU（交并比）: {iou * 100:.2f}%")
    
    # 2. 计算重叠区域边界框
    all_points = np.vstack((warped_corners.squeeze(), [[0, 0], [0, h2], [w2, h2], [w2, 0]]))
    x_min, y_min = np.floor(all_points.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_points.max(axis=0)).astype(int)
    
    # 重叠区域尺寸
    overlap_w = x_max - x_min
    overlap_h = y_max - y_min
    # print("重叠区域尺寸为h:{0},w:{1}".format(overlap_h, overlap_w))
    
    # 3. 初始化网格容器
    grid_rows, grid_cols = grid_size
    cell_width = max(1, overlap_w // grid_cols)
    cell_height = max(1, overlap_h // grid_rows)
    grid_points = {(i, j): [] for i in range(grid_rows) for j in range(grid_cols)}
    # 4. 将匹配点分配到网格
    inlier_counter = 0
    print("numa为{4}，numb为{5}，x选取范围为（{0},{1}）, y选取范围为（{2},{3}）".format(X_run - 5,X_run + 5,Y_run - 5,Y_run + 5,numa,numb))
    for i,m in enumerate(matches):
        if (inliers[i][0] == 0):
            continue
        kp_A = kp_1[m.trainIdx]
        kp_B = kp_2[m.queryIdx]

        # if ((numa-numb == 1) and((kp_B[0] - kp_A[0] > (X_run + 5) or kp_B[0]-kp_A[0] < (X_run -5)) or (kp_B[1]-kp_A[1]) > (-Y_run + 10) or (kp_B[1]-kp_A[1]) < (-Y_run - 10))):
        #     continue
        # # 正确
        # if ((numa-numb == -1) and ((kp_B[0] - kp_A[0] > (X_run + 5) or kp_B[0]-kp_A[0] < (X_run -5)) or (kp_B[1]-kp_A[1]) > (-Y_run + 10) or (kp_B[1]-kp_A[1]) < (-Y_run - 10))):
        #     continue
        # # 正确
        # if (numa - numb > 1) and ((kp_B[1]-kp_A[1]) > (-Y_run + 5) or (kp_B[1]-kp_A[1]) < (-Y_run - 5) or (kp_B[0] - kp_A[0]) > (X_run + 10) or (kp_B[0] - kp_A[0]) < (X_run -10)):
        #     continue
        # # 正确
        # if (numa - numb < -1) and ((kp_B[1]-kp_A[1]) > (-Y_run + 5) or (kp_B[1]-kp_A[1]) < (-Y_run - 5) or (kp_B[0] - kp_A[0]) > (X_run + 10) or (kp_B[0] - kp_A[0]) < (X_run -10)):
        #     continue

        # 获取图像2中的点坐标
        pt_img2 = np.array([kp_2[m.queryIdx]], dtype=np.float32).reshape(-1, 1, 2)
        
        # 投影到图像1坐标系
        pt_img1 = cv2.perspectiveTransform(pt_img2, H).squeeze()
        
        # 转换为重叠区域局部坐标
        local_x = pt_img1[0] - x_min
        local_y = pt_img1[1] - y_min
        # 计算网格索引
        grid_x = min(grid_rows - 1, int(local_y // cell_height))
        grid_y = min(grid_cols - 1, int(local_x // cell_width))
        # === 修复关键：检查网格索引是否越界 ===
        if grid_x < 0 or grid_x >= grid_rows or grid_y < 0 or grid_y >= grid_cols:
            continue  # 跳过超出网格范围的点
        # if (grid_x < 0 or grid_y < 0):
        #     print(grid_rows, grid_cols, cell_height, cell_width)
        # 存入对应网格
        grid_points[(grid_x, grid_y)].append(m)
        inlier_counter += 1
    
    print("全部的内点个数（没有筛选之前）matches:{3},inlier_counter:{0}, len(kp_2):{4}, gird的大小为col:{1}, row:{2}".format(inlier_counter, grid_cols, grid_rows,len(matches), len(kp_2)))
    
    # 5. 不随机选择P个匹配点 还是按照置信度选择
    selected_matches = []
    non_empty_grids = [coord for coord, points in grid_points.items() if points]
    # random.shuffle(non_empty_grids)  # 随机打乱网格顺序
    
    while len(selected_matches) < P and non_empty_grids:
        grid_coord = non_empty_grids.pop(0)
        points_in_grid = grid_points[grid_coord]
        
        if points_in_grid:
            # 选择网格内的一个点
            chosen_match = points_in_grid[0]
            selected_matches.append(chosen_match)
            
            # 从网格中移除已选点
            points_in_grid.remove(chosen_match)
            
            # 如果网格还有点，重新加入待选队列
            if points_in_grid:
                non_empty_grids.append(grid_coord)
    
    return selected_matches[:P]  # 确保不超过P个点

def calculate_geo_spacing(base_lat, base_lon):
    """
    计算水平拍摄时的经纬度间距
    
    参数:
        base_lat: 基准点纬度 (度)
        base_lon: 基准点经度 (度)
    
    返回:
        (lat_spacing, lon_spacing): 纬度和经度方向每移动100cm的度数变化
    """
    # 地球半径 (米)
    EARTH_RADIUS = 6371000  # 平均半径
    
    # 将拍摄距离转换为米 (100cm = 1m)
    distance = 1.0
    
    # 1. 计算纬度方向间距 (南北方向)
    # 纬度变化：1度 ≈ 111,000米 (恒定)
    lat_spacing = distance / (111000)  # 度
    
    # 2. 计算经度方向间距 (东西方向)
    # 经度变化随纬度增加而减小
    lat_rad = math.radians(base_lat)
    # 当前纬度的周长 = 赤道周长 × cos(纬度)
    circumference_at_lat = 2 * math.pi * EARTH_RADIUS * math.cos(lat_rad)
    # 1度经度对应的距离 (米)
    degrees_per_meter = 360 / circumference_at_lat
    lon_spacing = distance * degrees_per_meter
    
    return lat_spacing, lon_spacing


def gen_mark(args):
    '''
    生成mark图片，返回添加水印的函数
    '''
    # 字体宽度、高度
    is_height_crop_float = '.' in args["font-height-crop"]  # not good but work
    width = len(args["mark"]) * args["size"]
    if is_height_crop_float:
        height = round(args["size"] * float(args["font-height-crop"]))
    else:
        height = int(args["font-height-crop"])

    # 创建水印图片(宽度、高度)
    mark = Image.new(mode='RGBA', size=(width, height))

    # 生成文字
    draw_table = ImageDraw.Draw(im=mark)
    draw_table.text(xy=(0, 0),
                    text=args["mark"],
                    fill=args["color"],
                    font=ImageFont.truetype(args["font-family"],
                                            size=args["size"]))
    del draw_table

    # 裁剪空白
    mark = crop_image(mark)

    # 透明度
    set_opacity(mark, args["opacity"])

    def mark_im(im):
        ''' 在im图片上添加水印 im为打开的原图'''

        # 计算斜边长度
        c = int(math.sqrt(im.size[0] * im.size[0] + im.size[1] * im.size[1]))

        # 以斜边长度为宽高创建大图（旋转后大图才足以覆盖原图）
        mark2 = Image.new(mode='RGBA', size=(c, c))

        # 在大图上生成水印文字，此处mark为上面生成的水印图片
        y, idx = 0, 0
        while y < c:
            # 制造x坐标错位
            x = -int((mark.size[0] + args["space"]) * 0.5 * idx)
            idx = (idx + 1) % 2

            while x < c:
                # 在该位置粘贴mark水印图片
                mark2.paste(mark, (x, y))
                x = x + mark.size[0] + args["space"]
            y = y + mark.size[1] + args["space"]

        # 将大图旋转一定角度
        mark2 = mark2.rotate(args["angle"])

        # 在原图上添加大图水印
        if im.mode != 'RGBA':
            im = im.convert('RGBA')
        im.paste(mark2,  # 大图
                 (int((im.size[0] - c) / 2), int((im.size[1] - c) / 2)),  # 坐标
                 mask=mark2.split()[3])
        del mark2
        return im

    return mark_im  


def Show_Select_matching(src1, src2, kp1, kp2, matches):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]
    # print("开始生成可视化图片！")
    
    # 设置字体参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.2  # 减小字体大小以避免重叠
    font_color = (255, 0, 255)  # 洋红色
    font_thickness = 1
    
    # 设置dx/dy显示的字体参数（使用不同颜色区分）
    dxdy_font_scale = 0.3
    dx_color = (255, 255, 0)  # 蓝色表示dx
    dy_color = (0, 255, 255)  # 黄色表示dy
    
    for i in range(len(matches)):
        # 获取匹配点的坐标
        left = kp1[matches[i].queryIdx].pt
        right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
        
        # 绘制匹配连线
        cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0), 1)
        
        # 计算连线中点位置用于放置编号
        mid_x = int((left[0] + right[0]) / 2)
        mid_y = int((left[1] + right[1]) / 2)
        
        # 在连线旁边绘制编号
        cv2.putText(output, str(i), (mid_x + 1, mid_y - 1), font, font_scale, font_color, font_thickness)
        
        # 计算dx，dy（实际坐标差）
        dx = abs(left[0] - (right[0] - src1.shape[1])) # 注意调整右侧图像的x坐标
        dy = abs(left[1] - right[1])
        
        # 在连线下方显示dx和dy值
        dxdy_y_offset = 5  # 与编号的垂直偏移量
        cv2.putText(output, f"dx:{dx}", (mid_x + 1, mid_y + dxdy_y_offset), 
                   font, dxdy_font_scale, dx_color, font_thickness)
        cv2.putText(output, f"dy:{dy}", (mid_x + 1, mid_y + dxdy_y_offset + 6), 
                   font, dxdy_font_scale, dy_color, font_thickness)
    
    for i in range(len(matches)):
        left = kp1[matches[i].queryIdx].pt
        right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
        
        # 绘制特征点
        cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), -1)  # 左侧点：黄色实心
        cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), -1)   # 右侧点：绿色实心
    
    return output

def draw_matches(src1, src2, kp1, kp2, matches,inliers = None, max_matches_to_use = 0, drawing_type = 2):
    # print("src1 channels:", src1.shape[2] if len(src1.shape)>2 else 1)
    # print("src2 channels:", src2.shape[2] if len(src2.shape)>2 else 1)
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]
    inlier_counter = 0

    if drawing_type == 1:#DrawingType.ONLY_LINES
        print(len(kp1),len(kp2), len(matches))
        for i in range(len(matches)):
            # if (matches[i].queryIdx > len(kp1)):
            #     print(matches[i].queryIdx, matches[i].trainIdx)
            #     print(kp1[matches[i].queryIdx].pt, kp2[matches[i].trainIdx].pt)
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

    elif drawing_type == 2:#DrawingType.LINES_AND_POINTS
        inlier_counter = 0
        for i in range(len(matches)):
            # print("len(kp1) = {0},len(kp2) = {1}, len(matches) = {2}".format(len(kp1), len(kp2), len(matches)))
            if not inliers == None and inliers[i,0] == 0:
                continue
            # if (inlier_counter >= max_matches_to_use):
            #     break
            if (matches[i].queryIdx > len(kp1)):
                print("len(kp1) = {0},len(kp2) = {1}, len(matches) = {2}".format(len(kp1), len(kp2), len(matches)))
                print("matches[i].queryIdx = {0}, matches[i].trainIdx = {1}".format(matches[i].queryIdx, matches[i].trainIdx))
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))
            inlier_counter += 1

        inlier_counter = 0
        for i in range(len(matches)):
            if not inliers == None and inliers[i,0] == 0:
                continue
            # if (inlier_counter >= max_matches_to_use):
            #     break
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)
            inlier_counter += 1

    elif drawing_type == 3 or drawing_type == 4 or drawing_type == 5:#DrawingType.COLOR_CODED_POINTS_X | DrawingType.COLOR_CODED_POINTS_Y | DrawingType.COLOR_CODED_POINTS_XpY
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(matches)):
            left = kp1[matches[i].trainIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].queryIdx].pt, (src1.shape[1], 0)))

            if drawing_type == 3:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == 4:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == 5:
                colormap_idx = int((left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5))  # manhattan gradient

            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output
def CalculateCornerRange(pts2):
  
    min_x = sys.maxsize
    max_x = 0
    min_y = sys.maxsize
    max_y = 0
    
    if pts2[0][0] > max_x:
        max_x = pts2[0][0]
    if pts2[1][0] > max_x:
        max_x = pts2[1][0]
    if pts2[3][0] > max_x:
        max_x = pts2[3][0]
    if pts2[2][0] > max_x:
        max_x = pts2[2][0]

    if pts2[0][1] > max_y:
        max_y = pts2[0][1]
    if pts2[1][1] > max_y:
        max_y = pts2[1][1]
    if pts2[3][1] > max_y:
        max_y = pts2[3][1]
    if pts2[2][1] > max_y:
        max_y = pts2[2][1]

    if pts2[0][0] < min_x:
        min_x = pts2[0][0]
    if pts2[1][0] < min_x:
        min_x = pts2[1][0]
    if pts2[3][0] < min_x:
        min_x = pts2[3][0]
    if pts2[2][0] < min_x:
        min_x = pts2[2][0]

    if pts2[0][1] < min_y:
        min_y = pts2[0][1]
    if pts2[1][1] < min_y:
        min_y = pts2[1][1]
    if pts2[3][1] < min_y:
        min_y = pts2[3][1]
    if pts2[2][1] < min_y:
        min_y = pts2[2][1]

    return math.floor(min_x), math.ceil(max_x), math.floor(min_y), math.ceil(max_y)


def scaleAndCrop(img, gray=True):
    """将最后的大图，去掉黑边（通过阈值分割，锁定感兴趣区域，找到最小外接矩形）"""
    if gray:
        grey = img
    else:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
    # out = cv2.findContours(thresh, 1, 2)
    # cnt = out[0]
    x,y,w,h = cv2.boundingRect(grey)
    crop = img[y:y+h, x:x+w]
    return x, y, w, h, crop

def create_laplacian_pyramid(image, levels):
    # 创建拉普拉斯金字塔
    gaussian_pyramid = [image]
    jiou = []
    for i in range(levels):
        jiou.append(image.shape[1]%2) 
        jiou.append(image.shape[0]%2)    
        image = cv2.pyrDown(image) 
              
        gaussian_pyramid.append(image)

    laplacian_pyramid = [gaussian_pyramid[-1]]  # 最底层是高斯金字塔的最后一层
    for i in range(levels, 0, -1):
        size = (gaussian_pyramid[i].shape[1]*2-jiou[2*i-2], gaussian_pyramid[i].shape[0]*2-jiou[2*i-1])

        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i],dstsize=size)

        laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)

    return laplacian_pyramid
def laplacian_blending(images, levels):
    # 确保所有图像的尺寸一致
    images = [cv2.resize(img, (images[0].shape[1], images[0].shape[0])) for img in images]
    
    # 创建拉普拉斯金字塔
    laplacian_pyramids = [create_laplacian_pyramid(img, levels) for img in images]

    # 融合拉普拉斯金字塔
    blended_pyramid = []
    jiou = []
    for i in range(len(laplacian_pyramids[0])):
        blended_layer = np.zeros_like(laplacian_pyramids[0][i])  # 初始化为零
        weight_sum = np.zeros(blended_layer.shape)  # 用于权重和
        for j in range(len(laplacian_pyramids)):
            mask = (laplacian_pyramids[j][i] != 0)  # 只考虑非黑色部分
            blended_layer[mask] += laplacian_pyramids[j][i][mask]  # 叠加非黑色部分
            weight_sum[mask] += 1  # 计算权重和
        # 处理权重，避免除以零
        blended_layer = np.divide(blended_layer, weight_sum, where=(weight_sum != 0))
        blended_pyramid.append(blended_layer)
        
        # blended_layer = sum(pyramid[i] for pyramid in laplacian_pyramids)
        # blended_pyramid.append(blended_layer)
        if i!=0:
            jiou.append(blended_layer.shape[1]%2) 
            jiou.append(blended_layer.shape[0]%2)   
        print(blended_layer.shape)
    print(jiou)
    # 重构融合图像
    blended_image = blended_pyramid[0]
    for i in range(1, len(blended_pyramid)):
        size = (blended_image.shape[1]*2-jiou[2*i-2], blended_image.shape[0]*2-jiou[2*i-1])
        blended_image = cv2.pyrUp(blended_image,dstsize=size)
        print(blended_image.shape, blended_pyramid[i].shape)
        blended_image = cv2.add(blended_image, blended_pyramid[i])

    return blended_image

def add_mask(contain,contain_mask,corner):
    if contain_mask is None:
        contain_mask = np.zeros_like(contain)[:,:,0]
        
    contain_gray = cv2.cvtColor(contain, cv2.COLOR_BGR2GRAY)
    corner_mask = np.zeros_like(contain_gray)

    corner = corner.reshape((-1, 1, 2))
    cv2.fillPoly(corner_mask, [corner], (255,255,255))
    
    intersect = cv2.bitwise_and(contain_mask,corner_mask)#求交集 求整体区域
    contain_mask = cv2.bitwise_or(contain_mask, corner_mask)#求并集 重叠区域
    del contain_gray,corner_mask
    return contain_mask,intersect#取一个通道
def get_4_part(contain,image,intersect):
    img_c01 = cv2.bitwise_and(contain, contain, mask=intersect)
    img_c02 = cv2.subtract(contain, img_c01)
    # img_c02 = cv2.bitwise_and(image1, image1, mask=mask_c02)

    img_i01 = cv2.bitwise_and(image, image, mask=intersect)
    img_i02 = cv2.subtract(image, img_i01)

    return img_c01,img_c02,img_i01,img_i02

def get_overlap(result1,result2,CornersA,CornersB):
    #求两张图片的重叠区域的平均像素Iab,Iba和总像素N_overlap
    # print(copy1.shape,copy2.shape)
    result1_e = np.zeros_like(result1)
    result2_e = np.zeros_like(result2)
    
    CornersA = CornersA.reshape((-1, 1, 2))
    CornersB = CornersB.reshape((-1, 1, 2))

    cv2.fillPoly(result1_e, [CornersA], (255,255,255))
    cv2.fillPoly(result2_e, [CornersB], (255,255,255))
    
    overlap = cv2.bitwise_and(result1_e,result2_e)
    
    N_overlap = overlap.sum()/255
    Iab = np.sum(result1[overlap == 255])/N_overlap
    Iba = np.sum(result2[overlap == 255])/N_overlap

    del result1_e,result2_e
    return Iab,Iba,N_overlap

def get_gps_distance(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    lambda1 = math.radians(lon1)
    phi2 = math.radians(lat2)
    lambda2 = math.radians(lon2)
    R = 6371e3

    a = math.sin((phi2 - phi1) / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (
        math.sin((lambda2 - lambda1) / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c
def get_xy_distance(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    lambda1 = math.radians(lon1)
    phi2 = math.radians(lat2)
    lambda2 = math.radians(lon2)
    R = 6371000 * math.pi /180

    y = (lat2-lat1) * R

    x = (lon2-lon1) * math.cos((lat1 + lat2) / 2) * R
    return abs(x),abs(y)

def get_image_detail(file_path):
    """
    :param file_path: 输入图片路径
    :return: 图片的偏航角

    """
    try:
        # 定义字节模式 b 和 a，用于查找大疆EXIF数据的起始和结束标记
        b = b"\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e"
        a = b"\x3c\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20"
        # 打开图片文件，以二进制模式读取
        img = open(file_path, 'rb')
        # 初始化一个字节数组用于存储EXIF数据
        data = bytearray()
        # 初始化一个标志，用于判断是否已经找到EXIF数据的起始标记
        flag = False
        # 逐行读取图片文件内容
        for line in img.readlines():
            # 如果当前行包含EXIF数据的起始标记，则设置标志为True
            if a in line:
                flag = True
                # 如果标志为True，则将当前行添加到EXIF数据中
            if flag:
                data += line
                # 如果当前行包含EXIF数据的结束标记，则跳出循环
            if b in line:
                break
                # 如果提取到的EXIF数据不为空
        dj_data_dict = {}
        # 遍历过滤后的行，并提取键值对存入字典中
        if len(data) > 0:
            # 将字节数据解码为ASCII字符串
            data = str(data.decode('ascii'))
            # 过滤出包含drone-dji的行，并分割每行为键值对
            lines = list(filter(lambda x: 'drone-dji:' in x, data.split("\n")))
            # 初始化一个空字典用于存储提取到的数据
            for d in lines:
                d = d.strip()[10:]  # 去除每行的前后空格和'\n'字符，并从第10个字符开始处理（因为drone-dji:占据了前9个字符）
                k, v = d.split("=")  # 将当前行分割为键和值两部分
                dj_data_dict[k] = v  # 将键值对存入字典中
                
        GimbalRollDegree = float(dj_data_dict['GimbalRollDegree'].strip('" '))
        GimbalPitchDegree = float(dj_data_dict['GimbalPitchDegree'].strip('" '))
        GimbalYawDegree = float(dj_data_dict['GimbalYawDegree'].strip('" '))
        RelativeAltitude = float(dj_data_dict['RelativeAltitude'].strip('" '))
        return RelativeAltitude, GimbalRollDegree, GimbalPitchDegree, GimbalYawDegree
    except (KeyError, ValueError, OSError) as e:
        # 发生任何异常时，打印错误信息并返回None, None, None, None
        # print(f"Error encountered: {e}，读取无人机数据失败")
        return None, None, None, None

def calculate_geo_diff_simple(base_lat, base_lon, horizontal_distance_cm, vertical_distance_cm):
    """
    使用简单球面模型计算相邻图片的经纬度差异
    
    参数:
        base_lat (float): 基准点的纬度（度）
        base_lon (float): 基准点的经度（度）
        horizontal_distance_cm (float): 水平方向（东西向）地面距离（厘米）
        vertical_distance_cm (float): 垂直方向（南北向）地面距离（厘米）
    
    返回:
        delta_lat (float): 纬度差（度），正值表示向北
        delta_lon (float): 经度差（度），正值表示向东
    """
    # 将厘米转换为米
    horizontal_distance_m = horizontal_distance_cm / 100.0
    vertical_distance_m = vertical_distance_cm / 100.0
    
    # 地球平均半径（米），使用WGS84标准椭球体的近似值
    R = 6371000
    
    # 计算每度纬度对应的米数（大致恒定）
    meters_per_degree_lat = (math.pi * R) / 180.0
    
    # 计算每度经度对应的米数（随纬度变化）
    meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians(base_lat))
    
    # 计算经纬度差异
    delta_lat = vertical_distance_m / meters_per_degree_lat  # 南北方向影响纬度
    delta_lon = horizontal_distance_m / meters_per_degree_lon # 东西方向影响经度
    
    return delta_lat, delta_lon

def calculate_geo_diff_gdal(base_lat, base_lon, horizontal_distance_cm, vertical_distance_cm):
    """
    使用 GDAL 库精确计算相邻图片的经纬度差异（考虑地球椭球模型）
    
    参数:
        base_lat (float): 基准点的纬度（度）
        base_lon (float): 基准点的经度（度）
        horizontal_distance_cm (float): 水平方向（东西向）地面距离（厘米）
        vertical_distance_cm (float): 垂直方向（南北向）地面距离（厘米）
    
    返回:
        delta_lat (float): 纬度差（度），正值表示向北
        delta_lon (float): 经度差（度），正值表示向东
    """
    # 将厘米转换为米
    horizontal_distance_m = horizontal_distance_cm / 100.0
    vertical_distance_m = vertical_distance_cm / 100.0
    
    # 创建坐标转换对象（WGS84椭球体）
    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(4326)  # WGS84坐标系
    
    # 创建目标投影（UTM，根据基准点自动计算合适带号）
    utm_zone = int((base_lon + 180) / 6) + 1
    is_northern = base_lat >= 0
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(32600 + utm_zone if is_northern else 32700 + utm_zone)
    
    # 创建坐标转换
    transform_to_utm = osr.CoordinateTransformation(source_srs, target_srs)
    transform_to_wgs84 = osr.CoordinateTransformation(target_srs, source_srs)
    
    # 将基准点转换为投影坐标（米）
    x_base, y_base, _ = transform_to_utm.TransformPoint(base_lon, base_lat)
    
    # 计算相邻点的投影坐标
    x_adjacent = x_base + horizontal_distance_m  # 东方向
    y_adjacent = y_base + vertical_distance_m   # 北方向
    
    # 将相邻点转换回经纬度
    lon_adjacent, lat_adjacent, _ = transform_to_wgs84.TransformPoint(x_adjacent, y_adjacent)
    
    # 计算差异
    delta_lon = lon_adjacent - base_lon
    delta_lat = lat_adjacent - base_lat
    
    return delta_lat, delta_lon

def get_GSD(pix, f, height,scale):
    '''
    :param pix:像元大小（微米）
    :param f: 相机焦距（毫米）
    :param height: 无人机飞行高度（米）
    :return:米/像素
    '''
    
    GSD =  pix * height / (f * 1000 *scale)
    # print(pix,height,f,GSD)
    return GSD

def get_gps_angle(lat1, lon1, lat2, lon2):
    """
    计算两个 GPS 坐标之间的方位角（azimuth）。

    lat1, lon1 -- 第一个点的纬度和经度（度）    lat2, lon2 -- 第二个点的纬度和经度（度）

    方位角（度），从第一个点到第二个点的方向，相对于北方的角度
    """
    # 将经纬度从度转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 计算经度差
    d_lon = lon2_rad - lon1_rad

    # 计算方位角
    x = math.sin(d_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(d_lon)
    
    azimuth_rad = math.atan2(x, y)

    # 转换为度
    azimuth_deg = math.degrees(azimuth_rad)

    # 将角度范围调整到0到360度
    azimuth_deg = (azimuth_deg + 360) % 360

    return azimuth_deg

def Insert_angle(p,angle,radius):
    '''
    设计类似圆盘的东西，输入每个角度会得到一个左右radius度的一个扇形，代表已经获得了这样的匹配对
    '''
    # 计算角度范围
    lower_bound = (angle - radius) % 360
    upper_bound = (angle + radius) % 360

    # 更新数组中的值
    if lower_bound < upper_bound:
        # 范围没有跨越0度
        p[lower_bound:upper_bound+1] = [1] * (upper_bound - lower_bound+1)
    else:
        # 范围跨越了0度
        p[lower_bound:] = [1] * (359 - lower_bound+1)
        p[:upper_bound+1] = [1] * (upper_bound+1)

def get_SIFT_points(main_img, bounding_box, max_sift_number,method):

    img = main_img.copy() 

    # sift = cv2.xfeatures2d.SIFT_create()
    if method == 0 or method == 1:
        sift = cv2.SIFT_create(nfeatures = max_sift_number)
        kp, desc = sift.detectAndCompute(img, None)
        # surf = cv2.xfeatures2d.SURF_create(20)
        # kp, desc = surf.detectAndCompute(img, None)
        # orb = cv2.ORB_create(10000)
        # orb.setFastThreshold(0)
        # kp, desc = orb.detectAndCompute(img, None)
    elif method == 2:
        orb = cv2.ORB_create(10000)
        orb.setFastThreshold(0)
        kp, desc = orb.detectAndCompute(img, None)
    elif method == 3:
        surf = cv2.xfeatures2d.SURF_create(20)
        kp, desc = surf.detectAndCompute(img, None)

    if desc is not None and kp is not None:
        kp = kp[: min(len(kp), max_sift_number)]
        desc = desc[: min(len(kp), max_sift_number)]
    else:
        kp, desc = [], []
    return kp, desc
def get_ORB_points(main_img, bounding_box, max_sift_number):

    img = main_img.copy()
    
    orb = cv2.ORB_create(2000)
    orb.setFastThreshold(0)

    kp, desc = orb.detectAndCompute(img, None)

    kp = kp[: min(len(kp), max_sift_number)]
    desc = desc[: min(len(kp), max_sift_number)]
    return kp, desc

def get_matches(desc1, desc2, kp1, kp2, perc_next_match=0.8, perc_top_matches=0.5):

    bf = cv2.BFMatcher()
    # bf = cv2.DescriptorMatcher_create("FlannBased") 
    matches = bf.knnMatch(desc1, desc2, k=2)

    if matches is None or len(matches) == 0:
        return None

    if len(matches[0]) < 2:
        return None

    good = []
    for m in matches:

        if m[0].distance < perc_next_match * m[1].distance:
            good.append(m)

    sorted_matches = sorted(good, key=lambda x: x[0].distance)

    good = []

    number_of_good_matches = int(math.floor(len(sorted_matches) * perc_top_matches))
    good = sorted_matches[0:number_of_good_matches]

    matches = np.asarray(good)

    return matches
def get_orb_matches(desc1, desc2, kp1, kp2,wh, perc_next_match=0.8, perc_top_matches=0.5):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_all = matcher.match(desc1, desc2)

    kp1 = [keypoint_from_tuple_decode_analyse(p) for p in kp1]
    kp2 = [keypoint_from_tuple_decode_analyse(p) for p in kp2]
    matches_gms = matchGMS(wh, wh, kp1, kp2, matches_all, withScale=False, withRotation=False, thresholdFactor=6)
    

    return matches_gms
    


def estimate_base_transformations(pts1, pts2, tr_type):

    if tr_type == Transformation.translation:

        T = np.eye(3)
        mean_xys = np.mean(pts2 - pts1, axis=0)

        T[0, 2] = mean_xys[0]
        T[1, 2] = mean_xys[0]
        return T

    if tr_type == Transformation.similarity:

        T = cv2.estimateAffinePartial2D(pts1, pts2)

        if T is None or len(T.shape) <= 1:
            return None

        T = np.append(T, np.array([[0, 0, 1]]), axis=0)

        return T

    if tr_type == Transformation.affine:

        T = cv2.estimateAffine2D(pts1, pts2)

        if T is None or len(T.shape) <= 1:
            return None

        T = np.append(T, np.array([[0, 0, 1]]), axis=0)

        return T

    if tr_type == Transformation.homography:

        T = cv2.findHomography(pts1, pts2)[0]
        # T = cv2.getPerspectiveTransform(pts1,pts2)

        return T


def estimate_transformation_from_SIFT(
    desc1, desc2, kp1, kp2, transformation, perc_second, cores, wh, method
):

    # if multiplied by the key points of the first image, gives the key points of the second image
    # if T multiplied by the corners of the second image (in first image system) gives the corners of the first image
    if method == 0 or method ==3:
        matches = get_matches(desc1, desc2, kp1, kp2, perc_second)
    elif method == 2:
        matches = get_orb_matches(desc1, desc2, kp1, kp2, wh, perc_second)
        matches = np.array(matches)

    if len(matches.shape) == 1:
        first_matches = matches
    else:
        first_matches = matches[:, 0]

    if len(matches) == 0:
        return None, None, 0, None

    if type(kp1) == cv2.KeyPoint:
        src = np.float32(
            [[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] for m in first_matches]
        )
        dst = np.float32(
            [[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in first_matches]
        )
    else:
        src = np.float32(
            [[kp1[m.queryIdx][0], kp1[m.queryIdx][1]] for m in first_matches]
        )
        dst = np.float32(
            [[kp2[m.trainIdx][0], kp2[m.trainIdx][1]] for m in first_matches]
        )

    if transformation == Transformation.translation:

        # diff = np.mean(dst-src,axis=0)

        # T = np.eye(3)
        # T[0,2] = diff[0]
        # T[1,2] = diff[1]

        # return T, first_matches, 1, None

        T, masked = RANSAC.estimateTranslation(src, dst, cores)
        # T, masked = RANSAC.estimateTranslation(src,dst,1)

        if T is None:
            return None, None, 0, None

        return T, first_matches, np.sum(masked) / len(dst), masked

    if transformation == Transformation.similarity:
        # if len(src) < 4:
        #     return None, None, 0, None

        # T, masked = cv2.findHomography(
        #     src, dst, maxIters=500, confidence=0.99, method=cv2.RANSAC
        # )

        # if T is None:
        #     return None, None, 0, None


        # return T, first_matches, np.sum(masked) / len(dst), masked
    
        T, _ = cv2.estimateAffinePartial2D(src, dst)

        if T is None or T.shape != (2, 3):
            return None, None, 0, None

        T = np.append(T, np.array([[0, 0, 1]]), axis=0)

        masked = np.zeros((len(src), 1))
        for i, p_s in enumerate(src):
            new_dst = np.matmul(T, (p_s[0], p_s[1], 1))
            new_dst = new_dst / new_dst[2]
            if np.sqrt(np.sum((new_dst[:2] - dst[i]) ** 2)) <= 1.5:
                masked[i, 0] = 1
            else:
                masked[i, 0] = 0

        # s,theta,tx,ty = decompose_similarity(T)
        # T_new = build_transformation(Transformation.similarity,{'tr_x':tx,'tr_y':ty,'angle_theta':-theta,'scale_x':s,'center_rotation':(0,0)})
        # print(theta)
        # print(s)
        # print(T)
        # print(T_new)
        # T = T_new
        # corner_translations = get_corner_wise_transformations(T,matches,kp1,kp2,w,h)

        # return T, 1, corner_translations

        # new_dst = np.matmul(T,(src[0][0],src[0][1],1))
        # new_dst = new_dst/new_dst[2]
        # print(new_dst)
        # print(dst[0])

        return T, first_matches, 1, masked

    if transformation == Transformation.affine:

        # T, masked = cv2.estimateAffinePartial2D(dst, src , maxIters = 500, confidence = 0.99, refineIters = 5)
        T, _ = cv2.estimateAffine2D(src, dst)

        if T is None or T.shape != (2, 3):
            return None, None, 0, None

        T = np.append(T, np.array([[0, 0, 1]]), axis=0)

        # new_dst = np.matmul(T,(src[0][0],src[0][1],1))
        # new_dst = new_dst/new_dst[2]
        # print(new_dst)
        # print(dst[0])

        masked = np.zeros((len(src), 1))
        for i, p_s in enumerate(src):
            new_dst = np.matmul(T, (p_s[0], p_s[1], 1))
            new_dst = new_dst / new_dst[2]
            if np.sqrt(np.sum((new_dst[:2] - dst[i]) ** 2)) <= 1.5:
                masked[i, 0] = 1
            else:
                masked[i, 0] = 0

        return T, first_matches, 1, masked

    if transformation == Transformation.homography:

        if len(src) < 4:
            return None, None, 0, None

        T, masked = cv2.findHomography(
            src, dst, maxIters=500, confidence=0.99, method=cv2.RANSAC
        )

        # T = non_homogenouse_homography(src,dst)
        # masked = np.array([1]*len(dst))

        if T is None:
            return None, None, 0, None

        # return T, len(masked)/len(dst)

        # print(src[masked[:,0]==1,:])
        # print(dst[masked[:,0]==1,:])

        # new_dst = np.matmul(T,(src[0][0],src[0][1],1))
        # new_dst = new_dst/new_dst[2]
        # print(new_dst)
        # print(dst[0])

        # masked2 = np.zeros((len(src),1))
        # for i, p_s in enumerate(src):
        # 	 new_dst = np.matmul(T,(p_s[0],p_s[1],1))
        # 	 new_dst = new_dst/new_dst[2]
        # 	 if np.sqrt(np.sum((new_dst[:2] - dst[i])**2)) <= 1.5:
        # 		 masked2[i,0] = 1
        # 	 else:
        # 		 masked2[i,0] = 0

        # print(masked2-masked)

        return T, first_matches, np.sum(masked) / len(dst), masked


def estimate_transformation_from_Inliers(
    inliers, desc1, desc2, kp1, kp2, transformation
):

    src = np.float32([[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] for m in inliers])
    dst = np.float32([[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in inliers])

    if (
        len(src.shape) == 1
        or len(dst.shape) == 1
        or src.shape[0] < 4
        or dst.shape[0] < 4
    ):
        return None, None, None, None

    if transformation == Transformation.translation:

        diff = np.mean(dst - src, axis=0)

        T = np.eye(3)
        T[0, 2] = diff[0]
        T[1, 2] = diff[1]

        return T, inliers, 1, None

    if transformation == Transformation.similarity:

        T, _ = cv2.estimateAffinePartial2D(src, dst)

        if T is None or T.shape != (2, 3):
            return None, None, 0, None

        T = np.append(T, np.array([[0, 0, 1]]), axis=0)

        masked = np.zeros((len(src), 1))
        for i, p_s in enumerate(src):
            new_dst = np.matmul(T, (p_s[0], p_s[1], 1))
            new_dst = new_dst / new_dst[2]
            if np.sqrt(np.sum((new_dst[:2] - dst[i]) ** 2)) <= 0.5:
                masked[i, 0] = 1
            else:
                masked[i, 0] = 0

        return T, inliers, 1, masked

    if transformation == Transformation.affine:

        # T, masked = cv2.estimateAffinePartial2D(dst, src , maxIters = 500, confidence = 0.99, refineIters = 5)
        T, _ = cv2.estimateAffine2D(src, dst)

        if T is None or T.shape != (2, 3):
            return None, None, 0, None

        T = np.append(T, np.array([[0, 0, 1]]), axis=0)

        masked = np.zeros((len(src), 1))
        for i, p_s in enumerate(src):
            new_dst = np.matmul(T, (p_s[0], p_s[1], 1))
            new_dst = new_dst / new_dst[2]
            if np.sqrt(np.sum((new_dst[:2] - dst[i]) ** 2)) <= 1.5:
                masked[i, 0] = 1
            else:
                masked[i, 0] = 0

        return T, inliers, 1, masked

    if transformation == Transformation.homography:

        T, masked = cv2.findHomography(
            src, dst, maxIters=500, confidence=0.99, method=cv2.RANSAC
        )

        if T is None:
            return None, None, 0, None

        return T, inliers, np.sum(masked) / len(dst), masked


def select_SIFT_points(kp, desc, bounding_box):

    x1 = bounding_box[0]
    y1 = bounding_box[1]
    x2 = bounding_box[2]
    y2 = bounding_box[3]


def draw_SIFT_points_on_img(img, kp, desc):
    res = img.copy()
    res = cv2.drawKeypoints(
        img, kp, res, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return res


def warp_homography_and_stitch(img, frame_image, frame_size, H):

    tmp = cv2.warpPerspective(img, H, frame_size)

    frame_image[frame_image == 0] = tmp[frame_image == 0]

    return frame_image


def find_warp_homography_and_warp(pts1, pts2, img, frame_size):
    H = estimate_base_transformations(pts1, pts2, Transformation.homography)
    if H is None:
        return None
    tmp = cv2.warpPerspective(img, H, frame_size)

    return tmp


def show(img, window_name, w, h):

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w, h)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)


def keypoint_to_tuple_encode(kp):
    # return kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id
    return kp.pt[0], kp.pt[1]

def keypoint_from_tuple_decode_analyse(tpl):
    kp = cv2.KeyPoint(
        x=tpl[0],
        y=tpl[1],
        size=0,
    )
    return kp

def keypoint_from_tuple_decode(tpl):
    kp = cv2.KeyPoint(
        x=tpl[0][0],
        y=tpl[0][1],
        _size=tpl[1],
        _angle=tpl[2],
        _response=tpl[3],
        _octave=tpl[4],
        _class_id=tpl[5],
    )
    return kp


def pickle_matches(matches):

    if matches is None:
        return None

    if len(matches) == 0:
        return []

    p_matches = []

    for m in matches:
        p_matches.append((m.distance, m.trainIdx, m.queryIdx, m.imgIdx))

    return np.array(p_matches)


def get_matches_from_pickled(pickled_matches):

    if pickled_matches is None:
        return None

    if len(pickled_matches.shape) == 0 or pickled_matches.shape[0] == 0:
        return []

    matches = []

    for d, tid, qid, iid in pickled_matches:
        matches.append(cv2.DMatch(int(qid), int(tid), int(iid), d))

    return matches


def non_homogenouse_homography(pts_n, pts_i):

    A = []
    b = []

    pts_n = pts_n
    pts_i = pts_i

    for i, p_i in enumerate(pts_i):

        p_n = pts_n[i]

        A.append([p_n[0], p_n[1], 1, 0, 0, 0, 0, 0, 0])
        b.append(p_i[0])

        A.append([0, 0, 0, p_n[0], p_n[1], 1, 0, 0, 0])
        b.append(p_i[1])

        A.append([0, 0, 0, 0, 0, 0, p_n[0], p_n[1], 1])
        b.append(1)

    A = np.array(A)
    b = np.array(b)

    res = lsq_linear(A, b)
    X = res.x

    T = X.reshape((3, 3))
    # print(np.matmul(T,np.array([3,5,1])))
    return T


def decompose_similarity(T):

    s = math.sqrt(T[0, 0] ** 2 + T[0, 1] ** 2)
    theta = np.degrees(np.arctan2(-T[0, 1], T[0, 0]))
    t_x = T[0, 2]
    t_y = T[1, 2]

    return np.round(np.array([s, theta, t_x, t_y]), 4)


def normalize_key_points(kp, w, h, initial):

    new_kp = []

    for p in kp:

        p_new = cv2.KeyPoint(
            x=(p.pt[0] - w / 2) / (w),
            y=(p.pt[1] - h / 2) / (h),
            _size=p.size,
            _angle=p.angle,
            _response=p.response,
            _octave=p.octave,
            _class_id=p.class_id,
        )
        # p_new = cv2.KeyPoint(x=p.pt[0]-w/2,y=p.pt[1]-h/2,_size=p.size, _angle=p.angle, _response=p.response, _octave=p.octave, _class_id=p.class_id)
        # p_new = cv2.KeyPoint(x=p.pt[0]-w/2+initial[0],y=p.pt[1]-h/2+initial[1],_size=p.size, _angle=p.angle, _response=p.response, _octave=p.octave, _class_id=p.class_id)
        # p_new = cv2.KeyPoint(x=p.pt[0]+initial[0],y=p.pt[1]+initial[1],_size=p.size, _angle=p.angle, _response=p.response, _octave=p.octave, _class_id=p.class_id)
        # p_new = cv2.KeyPoint(x=p.pt[0]/w-0.5+initial[0],y=p.pt[1]/h-0.5-initial[1],_size=p.size, _angle=p.angle, _response=p.response, _octave=p.octave, _class_id=p.class_id)
        new_kp.append(p_new)

    return new_kp


def get_best_single_good_match(T, matches, kp1, kp2):

    if len(matches.shape) == 1:
        first_matches = matches
    else:
        first_matches = matches[:, 0]

    error = sys.maxsize
    best_match = None

    for m in first_matches:
        p1 = np.array([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], 1])
        p2 = np.array([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1], 1])

        p1_transformed = np.matmul(T, p1)
        e = math.sqrt(np.mean((p1_transformed - p2) ** 2))

        if e < error:
            best_match = m
            error = e

    return best_match


def get_corner_wise_transformations(T, matches, kp1, kp2, w, h):

    best_m = get_best_single_good_match(T, matches, kp1, kp2)

    p1 = np.array([kp1[best_m.queryIdx].pt[0], kp1[best_m.queryIdx].pt[1]])
    p2 = np.array([kp2[best_m.trainIdx].pt[0], kp2[best_m.trainIdx].pt[1]])

    T_left_UL = np.eye(3)
    T_right_UL = np.eye(3)

    T_left_UR = np.eye(3)
    T_right_UR = np.eye(3)

    T_left_LR = np.eye(3)
    T_right_LR = np.eye(3)

    T_left_LL = np.eye(3)
    T_right_LL = np.eye(3)

    T_right_UL[0, 2] = p1[0]
    T_right_UL[1, 2] = p1[1]
    T_left_UL[0, 2] = -p2[0]
    T_left_UL[1, 2] = -p2[1]

    T_right_UR[0, 2] = p1[0] - w
    T_right_UR[1, 2] = p1[1]
    T_left_UR[0, 2] = -p2[0] + w
    T_left_UR[1, 2] = -p2[1]

    T_right_LR[0, 2] = p1[0] - w
    T_right_LR[1, 2] = p1[1] - h
    T_left_LR[0, 2] = -p2[0] + w
    T_left_LR[1, 2] = -p2[1] + h

    T_right_LL[0, 2] = p1[0]
    T_right_LL[1, 2] = p1[1] - h
    T_left_LL[0, 2] = -p2[0]
    T_left_LL[1, 2] = -p2[1] + h

    list_Ts = {}
    list_Ts["UL"] = [T_left_UL, T_right_UL]
    list_Ts["UR"] = [T_left_UR, T_right_UR]
    list_Ts["LR"] = [T_left_LR, T_right_LR]
    list_Ts["LL"] = [T_left_LL, T_right_LL]

    return list_Ts


def build_transformation(transformation_type, params):

    T_1 = np.eye(3)

    if transformation_type == Transformation.similarity:
        t_x = params["tr_x"]
        t_y = params["tr_y"]
        rotation_angle = params["angle_theta"]
        uniform_scale = params["scale_x"]
        center_rotation = params["center_rotation"]

        rot_mat = cv2.getRotationMatrix2D(
            center_rotation, rotation_angle, uniform_scale
        )

        T_1[0:2, :] = rot_mat[0:2, :]
        T_1[0, 2] += t_x
        T_1[1, 2] += t_y

    return T_1


def histogram_equalization(img):

    if len(img.shape) == 2:
        channel_0 = cv2.equalizeHist(img[:, :])

        img[:, :] = channel_0
    else:
        channel_0 = cv2.equalizeHist(img[:, :, 0])
        channel_1 = cv2.equalizeHist(img[:, :, 1])
        channel_2 = cv2.equalizeHist(img[:, :, 2])

        img[:, :, 0] = channel_0
        img[:, :, 1] = channel_1
        img[:, :, 2] = channel_2

    return img


def get_full_transformation(src, dst):

    A = []
    b = []

    for i, p1 in enumerate(src):

        p2 = dst[i]

        A.append([p1[0], p1[1], 1, 0, 0, 0, 0, 0, 0])
        b.append(p2[0])

        A.append([0, 0, 0, p1[0], p1[1], 1, 0, 0, 0])
        b.append(p2[1])

        A.append([0, 0, 0, 0, 0, 0, p1[0], p1[1], 1])
        b.append(1)

    A = np.array(A)
    b = np.array(b)

    res = lsq_linear(A, b)
    X = res.x

    T = X.reshape((3, 3))

    return T


def get_Similarity_Affine(src, dst):

    A = []
    b = []

    for i, p1 in enumerate(src):

        p2 = dst[i]

        A.append([p1[0], p1[1], 1, 0, 0, 0])
        b.append(p2[0])

        A.append([0, 0, 0, p1[0], p1[1], 1])
        b.append(p2[1])

    A = np.array(A)
    b = np.array(b)

    res = lsq_linear(A, b, tol=1e-10)
    X = res.x

    # X = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T,A)),A.T),b)

    T = X.reshape((2, 3))

    return T


def Jsonify(transformation_dict):

    jsonified = {}

    for img1 in transformation_dict:

        if img1 not in jsonified:
            jsonified[img1] = {}

        for img2 in transformation_dict[img1]:
            T = transformation_dict[img1][img2][0]
            matches = transformation_dict[img1][img2][1]
            perc_inlier = transformation_dict[img1][img2][2]
            inlier = transformation_dict[img1][img2][3]
            bins = transformation_dict[img1][img2][4]

            # T = T.tolist()
            if isinstance(T, np.ndarray):
                T = T.tolist()  # 转换为列表
            elif isinstance(T, list):
                pass  # 已是列表，无需转换
            inlier = inlier.tolist()
            matches = pickle_matches(matches).tolist()

            if bins is not None:
                new_bins = {}
                for i in bins:
                    new_bins[i] = pickle_matches(bins[i]).tolist()

                bins = new_bins

            jsonified[img1][img2] = (T, matches, perc_inlier, inlier, bins)

    return jsonified


def Unjsonify(jsonified_dict):
    unjsonified = {}

    for img1 in jsonified_dict:

        if img1 not in unjsonified:
            unjsonified[img1] = {}

        for img2 in jsonified_dict[img1]:
            T = jsonified_dict[img1][img2][0]
            matches = jsonified_dict[img1][img2][1]
            perc_inlier = jsonified_dict[img1][img2][2]
            inlier = jsonified_dict[img1][img2][3]
            bins = jsonified_dict[img1][img2][4]

            T = np.array(T)
            inlier = np.array(inlier)
            matches = get_matches_from_pickled(np.array(matches))

            if bins is not None:
                new_bins = {}
                for i in bins:
                    new_bins[i] = get_matches_from_pickled(np.array(bins[i]))

                bins = new_bins

            unjsonified[img1][img2] = (T, matches, perc_inlier, inlier, bins)

    return unjsonified


def generate_n_distinc_colors(n):

    list_colors = [
        (160, 82, 45),
        (47, 79, 79),
        (0, 0, 128),
        (255, 69, 0),
        (0, 206, 209),
        (255, 255, 0),
        (199, 21, 133),
        (255, 0, 255),
        (240, 230, 140),
        (100, 149, 237),
        (255, 192, 203),
        (160, 82, 45),
        (47, 79, 79),
        (0, 0, 128),
        (255, 69, 0),
        (0, 206, 209),
        (255, 255, 0),
        (199, 21, 133),
        (255, 0, 255),
        (240, 230, 140),
        (100, 149, 237),
        (255, 192, 203),
        (160, 82, 45),
        (47, 79, 79),
        (0, 0, 128),
        (255, 69, 0),
        (0, 206, 209),
        (255, 255, 0),
        (199, 21, 133),
        (255, 0, 255),
        (240, 230, 140),
        (100, 149, 237),
        (255, 192, 203),
    ]

    return list_colors[:n]


def get_GCP_sizes(dataset, method):

    if dataset == "DLLFN" and method == "BNDL-ADJ":
        s1 = 30
        s2 = 40
    elif dataset == "DLLFN" and method == "MEGASTITCH-AFF-BNDL-ADJ-OLDIN":
        s1 = 40
        s2 = 50
    elif dataset == "DLLFN" and method == "MEGASTITCH-SIM":
        s1 = 40
        s2 = 50
    if dataset == "DSEFN" and method == "BNDL-ADJ":
        s1 = 30
        s2 = 40
    elif dataset == "DSEFN" and method == "MEGASTITCH-AFF-BNDL-ADJ-OLDIN":
        s1 = 30
        s2 = 40
    elif dataset == "DSEFN" and method == "MEGASTITCH-SIM":
        s1 = 40
        s2 = 50
    elif dataset == "GCD" or dataset == "GRG":
        s1 = 20
        s2 = 30
    else:
        s1 = 30
        s2 = 40

    return s1, s2
