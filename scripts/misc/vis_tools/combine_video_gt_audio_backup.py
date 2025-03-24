import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, clips_array, TextClip, CompositeVideoClip
from os.path import join
import numpy as np

# 定义文件路径
video_path_gt = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2/smpl_fit/results_smplfitting/"
video_path1 = "results/emgpt/debug--Pretrain_Beat2_After_Amass/samples_2024-09-21-22-29-42/smpl_fit/results_smplfitting/"
video_path2 = "results/emgpt/debug--Pretrain_Beat2_And_Amass/samples_2024-09-21-22-45-02/smpl_fit/results_smplfitting/"
video_path3 = "results/emgpt/debug--Pretrain_Beat2_And_Amass/samples_2024-09-22-14-09-27/smpl_fit/results_smplfitting/"
audio_path = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2/beat2_audio_segmented/"


for file_name in os.listdir(video_path3):

    if file_name.endswith('.mp4')==False:
        continue
    # 要处理的文件名
    file_name_gt = file_name.replace('_out','')
    file_name_out = file_name
    audio_file_name = file_name.replace('_out','').replace('_mesh','').replace('mp4','wav')
    output_file = join('results/visualization', file_name)

    if os.path.exists(output_file):
        print(f"mp4 is already processed: {output_file}")
        continue
    # 构造完整路径
    video_file_gt = join(video_path_gt, file_name_gt)
    video_file1 = join(video_path1, file_name_out)
    video_file2 = join(video_path2, file_name_out)
    video_file3 = join(video_path3, file_name_out)
    audio_file = join(audio_path, audio_file_name)


    # 加载视频和音频
    try:
        video_clip_gt = VideoFileClip(video_file_gt)
        video_clip1 = VideoFileClip(video_file1)
        video_clip2 = VideoFileClip(video_file2)
        video_clip3 = VideoFileClip(video_file3)
        audio_clip = AudioFileClip(audio_file)
    except:
        continue

    # 确保两个视频时长一致（取较短时长）
    # min_duration = min(video_clip_gt.duration, video_clip1.duration, video_clip2.duration, video_clip3.duration)
    min_duration = np.median([(video_clip_gt.duration, video_clip1.duration, video_clip2.duration, video_clip3.duration)])
    video_clip_gt = video_clip_gt.subclip(0, min_duration)
    video_clip1 = video_clip1.subclip(0, min_duration)
    video_clip2 = video_clip2.subclip(0, min_duration)
    video_clip3 = video_clip3.subclip(0, min_duration)

    # 添加文字注释
    txt_clip_gt = TextClip("groundtruth", fontsize=30, color='black', font='Arial').set_position(("center", "bottom")).set_duration(min_duration)
    txt_clip1 = TextClip("stage1:AMASS, stage2:BEAT2, epoch_num:139", fontsize=20, color='black', font='Arial').set_position(("center", "bottom")).set_duration(min_duration)
    txt_clip2 = TextClip("stage1:AMASS, stage2:AMASS+BEAT2, epoch_num:19", fontsize=20, color='black', font='Arial').set_position(("center", "bottom")).set_duration(min_duration)
    txt_clip3 = TextClip("stage1:AMASS, stage2:AMASS+BEAT2, epoch_num:29", fontsize=20, color='black', font='Arial').set_position(("center", "bottom")).set_duration(min_duration)

    # 在视频下方添加文字
    video_with_txt_gt = CompositeVideoClip([video_clip_gt, txt_clip_gt])
    video_with_txt1 = CompositeVideoClip([video_clip1, txt_clip1])
    video_with_txt2 = CompositeVideoClip([video_clip2, txt_clip2])
    video_with_txt3 = CompositeVideoClip([video_clip3, txt_clip3])


    # 左右拼接视频
    # final_video = clips_array([[video_with_txt_gt, video_with_txt1, video_with_txt2, video_with_txt3]])
    # 使用 clips_array 创建四宫格布局
    final_video = clips_array([
        [video_with_txt_gt, video_with_txt1],  # 第一行两个视频
        [video_with_txt2, video_with_txt3]     # 第二行两个视频
    ])

    # 给拼接视频添加音频
    final_video = final_video.set_audio(audio_clip)

    # 输出文件
    final_video.write_videofile(output_file, codec="libx264", audio_codec="aac")

    # 释放资源
    video_clip1.close()
    video_clip2.close()
    audio_clip.close()