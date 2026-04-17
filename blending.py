import datetime
import multiprocessing
import os
from datetime import time

import cv2
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    start = datetime.datetime.now()
    folder_path = "/home/hipeson/lio/Base/Result/19.9-2-ALL/Result"
    # folder_path = 'overlap'

    image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    h, w, _ = img.shape
    print(h, w)
    # size = [1273,1308]
    # H = np.array([[0.9215522281032663, -0.09308442798578087, 466.4841474591958], [0.11527724625475594, 0.8404022743304578, -377.79679650703537], [0.0, 0.0, 1.0]])
    # img = cv2.warpPerspective(img,H,size)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    blended_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    accumulated_rgb = np.zeros((h, w, 3), dtype=np.float32)
    accumulated_mask = np.zeros((h, w), dtype=np.float32)



    #
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        # 使用OpenCV读取图像
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        rgb = img[:, :, :3].astype(np.float32)  # (w, h, 3)
        alpha = img[:, :, 3]
        mask = np.zeros((h, w), dtype=np.float32)
        mask[alpha > 0] = 1
        # alpha = img[ :, :, 3] / 255.0
        accumulated_rgb += rgb * mask[:, :, np.newaxis]
        accumulated_mask += mask

    # 防止除以0，将 Alpha 累积中为 0 的部分设置为 1
    blended_alpha = np.zeros_like(accumulated_mask, dtype=np.uint8)
    blended_alpha[accumulated_mask > 0] = 255

    accumulated_alpha = np.maximum(accumulated_mask, 1e-6)  # 避免为 0
    # 计算最终加权后的 RGB
    blended_rgb = accumulated_rgb / accumulated_alpha[:, :, None]
    # 创建 Alpha 通道，透明度大于 0 的部分设置为 255，其他为 0

    # 将结果转换为 uint8 格式
    blended_rgba[:, :, :3] = np.clip(blended_rgb, 0, 255).astype(np.uint8)  # 确保 RGB 值在 0-255 之间
    blended_rgba[:, :, 3] = blended_alpha  # Alpha 通道

    cv2.imwrite('blended_image.png', blended_rgba)
    end = datetime.datetime.now()
    print(end - start)