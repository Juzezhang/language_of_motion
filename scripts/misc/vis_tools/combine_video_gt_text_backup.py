import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, clips_array, TextClip, CompositeVideoClip
from os.path import join
import numpy as np
from collections import defaultdict
import csv
import pathlib
import codecs as cs

# 定义文件路径
# audio_path = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2_original/raw_data/beat_v2.0.0/beat_english_v2.0.0/wave16k/"

text_path = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/AMASS/texts/"



# 定义视频路径列表
video_path_list = [
    "/nas/nas_38/AI-being/libin/MDM_T2M_npy/T2M_GPT/results_smplfitting/",
    "/nas/nas_38/AI-being/libin/MDM_T2M_npy/MDM/results_smplfitting/",
    "/nas/nas_32/AI-being/zhangjz/exp_motion/paper_result/videos/t2m/",
]

save_path = "/nas/nas_32/AI-being/zhangjz/exp_motion/paper_result/videos/t2m_comparison/"


# 遍历文件夹，处理每个视频
for file_name in os.listdir(video_path_list[0]):  # 假设所有路径中的文件名相同，选择最后一个路径
    if not file_name.endswith('.mp4'):
        continue

    if file_name.startswith("M"):

        pass

    # 要处理的文件名
    file_name_out = file_name

    cleaned_file_name_new = file_name

    text_name = file_name.replace('.mp4', '.txt')


    with cs.open(join(text_path,text_name)) as f:
        text_data = []
        flag = False
        lines = f.readlines()

        line = lines[0]
        text_dict = {}
        line_split = line.strip().split('#')
        caption = line_split[0]
        t_tokens = line_split[1].split(' ')



    with cs.open(join(text_path,'M' + text_name)) as f:
        text_data = []
        flag = False
        lines = f.readlines()

        line = lines[0]
        text_dict = {}
        line_split = line.strip().split('#')
        caption_mirror = line_split[0]
        t_tokens = line_split[1].split(' ')


    output_file = join(save_path, file_name)

    if os.path.exists(output_file):
        print(f"mp4 is already processed: {output_file}")
        continue

    # 构造完整路径
    # video_file_gt = join(video_path_gt, file_name_gt)
    # video_files = [join(path, file_name_out) for path in video_path_list]
    video_files = [join(video_path_list[0], file_name_out),
                   join(video_path_list[1], 'M' + file_name_out),
                   ]

    # 加载视频和音频
    try:
        # video_clip_gt = VideoFileClip(video_file_gt)
        video_clips = [VideoFileClip(video_file) for video_file in video_files]
    except:
        continue

    # 确保所有视频时长一致（取中位数时长）
    durations = [clip.duration for clip in video_clips]
    min_duration = np.median(durations)

    # 截取每个视频到中位数时长
    # video_clip_gt = video_clip_gt.subclip(0, min_duration)
    video_clips = [clip.subclip(0, min_duration) for clip in video_clips]

    # 添加文字注释
    txt_clips = [TextClip(caption, fontsize=20, color='black', font='Arial').set_position(("center", "bottom")).set_duration(min_duration),
                 TextClip(caption_mirror, fontsize=20, color='black', font='Arial').set_position(("center", "bottom")).set_duration(min_duration),
                 ]
    # txt_clips = [
    #     TextClip(f"Video {i+1}: epoch_num", fontsize=20, color='black', font='Arial').set_position(("center", "bottom")).set_duration(min_duration)
    #     for i in range(len(video_clips))
    # ]

    # 在视频下方添加文字
    # video_with_txt_gt = CompositeVideoClip([video_clip_gt, txt_clip_gt])
    video_with_txt_clips = [CompositeVideoClip([clip, txt_clip]) for clip, txt_clip in zip(video_clips, txt_clips)]

    # 将groundtruth和其他视频拼接成四宫格（根据视频数量动态布局）
    video_grid = video_with_txt_clips
    num_cols = 3
    num_rows = int(np.ceil(len(video_grid) / num_cols))

    video_grid_layout = [video_grid[i * num_cols:(i + 1) * num_cols] for i in range(num_rows)]

    # 拼接视频成网格布局
    final_video = clips_array(video_grid_layout)

    # 输出文件
    final_video.write_videofile(output_file, codec="libx264", audio_codec="aac")

    # 释放资源
    for clip in video_clips:
        clip.close()