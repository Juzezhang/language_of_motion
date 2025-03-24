import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

# 定义路径
input_folder = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2/output_videos_with_audio/smpl_fit/"
output_folder = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2/output_videos_with_audio/smpl_fit_merged/"

# 创建输出文件夹，如果不存在
os.makedirs(output_folder, exist_ok=True)

# 获取文件列表并按前缀分组
file_dict = {}

for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        # 获取前缀，例如 '000000' from '000000_000_mesh.mp4'
        prefix = filename.split('_')[0]
        if prefix not in file_dict:
            file_dict[prefix] = []
        file_dict[prefix].append(filename)

# 遍历每组前缀，合并视频
for prefix, file_list in file_dict.items():
    # 确保文件按顺序合并
    file_list.sort()

    video_clips = []
    for file in file_list:
        video_path = os.path.join(input_folder, file)
        video_clip = VideoFileClip(video_path)
        video_clips.append(video_clip)

    # 合并视频
    if len(video_clips) > 0:
        final_clip = concatenate_videoclips(video_clips)

        # 输出合并后的文件
        output_file = os.path.join(output_folder, f"{prefix}.mp4")
        final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

        # 关闭视频资源
        for clip in video_clips:
            clip.close()

print("视频合并完成！")