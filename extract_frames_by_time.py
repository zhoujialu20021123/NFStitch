import cv2
import os

def extract_frames_by_time(video_path, output_folder, frames_per_second=1):
    """
    按时间间隔抽帧（每秒抽取指定帧数）
    :param video_path: 视频文件路径
    :param output_folder: 输出文件夹路径
    :param frames_per_second: 每秒要抽取的帧数
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return
    
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频帧率：{fps} FPS")
    
    # 计算抽帧间隔（每隔多少帧抽一帧）
    frame_interval = max(1, int(fps / frames_per_second))
    print(f"抽帧间隔：每 {frame_interval} 帧抽取一帧")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"抽帧完成！每秒抽取 {frames_per_second} 帧，共保存 {saved_count} 帧")
    print(f"预计视频时长：{frame_count/fps:.2f} 秒")

# 使用示例
if __name__ == "__main__":
    video_path = "/data/zhou/Wheat/单株视频（2.4-21.6米 速度25mm s-1）.mp4"
    output_folder = "/data/zhou/Wheat/frame_alone/"
    extract_frames_by_time(video_path, output_folder, frames_per_second=1)  # 每秒抽2帧