import os
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from os.path import join

# Input paths
video_path = "/nas/nas_32/AI-being/zhangjz/social_motion/experiments/cache_humintbm3_batch_epoch469_joint_selected/results_smplfitting/"
text_path = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets_heng/humintbm3/texts/"
save_path = "/nas/nas_32/AI-being/zhangjz/social_motion/experiments/cache_humintbm3_batch_epoch469_joint_selected_quality/"

# Ensure output directory exists
os.makedirs(save_path, exist_ok=True)

# Process each video file
for file_name in sorted(os.listdir(video_path)):

    # video_file = join(video_path, file_name, 'person_0_mesh.mp4')
    video_file = join(video_path, f"{file_name}.mp4")

    # Output file path
    output_file = join(save_path, f"{file_name}.mp4")

    if os.path.exists(output_file):
        print(f"Video file found: {output_file}")
        continue
    txt_name = file_name
    txt_name = txt_name.replace('_prompt_0', '')
    txt_name = txt_name.replace('_prompt_1', '')
    txt_name = txt_name.replace('_prompt_2', '')
    txt_name = txt_name.replace('_prompt_3', '')
    text_file = join(text_path, f"{txt_name}.txt")

    if not os.path.exists(video_file):
        print(f"Video file not found: {video_file}")
        continue

    if not os.path.exists(text_file):
        print(f"Text file not found: {text_file}")
        continue

    # Read text caption from the corresponding text file
    with open(text_file, 'r') as f:
        line = f.readline().strip()
        caption = line.split('#')[0]  # Get the first part before '#'

    # Load the video
    try:
        video_clip = VideoFileClip(video_file)
    except Exception as e:
        print(f"Error loading video {video_file}: {e}")
        continue

    # Create a TextClip with automatic line wrapping
    txt_clip = TextClip(
        caption,
        fontsize=20,
        color='black',
        # 若有中文且系统中已安装对应字体，可指定: font='SimHei'
        method='caption',        # 关键参数：开启自动换行
        size=(600, None),        # 文本区域宽度 600 像素，高度不固定
        align='center'           # 对齐方式，可选 'center', 'west', 'east'
    ).set_position(("center", "bottom")).set_duration(video_clip.duration)

    # Combine video with text
    video_with_text = CompositeVideoClip([video_clip, txt_clip])



    # Write the final video to a file
    video_with_text.write_videofile(output_file, codec="libx264", audio_codec="aac")

    # Close the video clip
    video_clip.close()
