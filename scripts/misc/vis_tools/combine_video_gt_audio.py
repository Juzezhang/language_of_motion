import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, clips_array, TextClip, CompositeVideoClip
from os.path import join
import numpy as np

# 定义文件路径
video_path_gt = "/nas/nas_32/AI-being/zhangjz/exp_motion/paper_result/quality_comparison/gt/videos/"
audio_path = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2_original/beat_v2.0.0/beat_english_v2.0.0/wave16k/"

# 定义视频路径列表
video_path_list = [
    "results/emgpt/debug--Pretrain_Beat2_After_Amass/samples_2024-09-21-22-29-42/smpl_fit/results_smplfitting/",
    "results/emgpt/debug--Pretrain_Beat2_And_Amass/samples_2024-09-21-22-45-02/smpl_fit/results_smplfitting/",
    "results/emgpt/debug--Pretrain_Beat2_And_Amass/samples_2024-09-22-14-09-27/smpl_fit/results_smplfitting/",
]

# 遍历文件夹，处理每个视频
for file_name in os.listdir(video_path_list[-1]):  # 假设所有路径中的文件名相同，选择最后一个路径
    if not file_name.endswith('.mp4'):
        continue

    # 要处理的文件名
    file_name_gt = file_name.replace('_out', '')
    file_name_out = file_name
    audio_file_name = file_name.replace('_out', '').replace('_mesh', '').replace('mp4', 'wav')
    output_file = join('results/visualization', file_name)

    if os.path.exists(output_file):
        print(f"mp4 is already processed: {output_file}")
        # continue

    # 构造完整路径
    video_file_gt = join(video_path_gt, file_name_gt)
    video_files = [join(path, file_name_out) for path in video_path_list]
    audio_file = join(audio_path, audio_file_name)

    # 加载视频和音频
    try:
        video_clip_gt = VideoFileClip(video_file_gt)
        video_clips = [VideoFileClip(video_file) for video_file in video_files]
        audio_clip = AudioFileClip(audio_file)
    except:
        continue

    # 确保所有视频时长一致（取中位数时长）
    durations = [video_clip_gt.duration] + [clip.duration for clip in video_clips]
    min_duration = np.median(durations)

    # 截取每个视频到中位数时长
    video_clip_gt = video_clip_gt.subclip(0, min_duration)
    video_clips = [clip.subclip(0, min_duration) for clip in video_clips]

    # 添加文字注释
    txt_clip_gt = TextClip("groundtruth", fontsize=30, color='black', font='Arial').set_position(("center", "bottom")).set_duration(min_duration)
    txt_clips = [
        TextClip(f"Video {i+1}: epoch_num", fontsize=20, color='black', font='Arial').set_position(("center", "bottom")).set_duration(min_duration)
        for i in range(len(video_clips))
    ]

    # 在视频下方添加文字
    video_with_txt_gt = CompositeVideoClip([video_clip_gt, txt_clip_gt])
    video_with_txt_clips = [CompositeVideoClip([clip, txt_clip]) for clip, txt_clip in zip(video_clips, txt_clips)]

    # 将groundtruth和其他视频拼接成四宫格（根据视频数量动态布局）
    video_grid = [video_with_txt_gt] + video_with_txt_clips
    num_cols = 2
    num_rows = int(np.ceil(len(video_grid) / num_cols))

    video_grid_layout = [video_grid[i * num_cols:(i + 1) * num_cols] for i in range(num_rows)]

    # 拼接视频成网格布局
    final_video = clips_array(video_grid_layout)

    # 给拼接视频添加音频
    final_video = final_video.set_audio(audio_clip)

    # 输出文件
    final_video.write_videofile(output_file, codec="libx264", audio_codec="aac")

    # 释放资源
    for clip in video_clips:
        clip.close()
    audio_clip.close()