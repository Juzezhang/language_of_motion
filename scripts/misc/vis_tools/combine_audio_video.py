import os
import moviepy.editor as mp

# Define paths
video_folder = "/nas/nas_32/AI-being/zhangjz/exp_motion/paper_result/videos/editable/"
audio_folder = "/nas/nas_32/AI-being/zhangjz/exp_motion/paper_result/videos/editable/"
output_folder = "/nas/nas_32/AI-being/zhangjz/exp_motion/paper_result/videos/editable/videos_audio/"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process video files
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        output_file = os.path.join(output_folder, video_file)
        if os.path.exists(output_file):
            continue
        # Remove "M" prefix and "_mesh" suffix from video filename for matching
        # cleaned_video_name = video_file.lstrip("M").replace("_mesh", "")
        # cleaned_video_name = video_file.lstrip("M").replace("_out", "").replace("_mesh", "")
        cleaned_video_name = video_file

        # Construct corresponding audio filename
        audio_file = cleaned_video_name.replace(".mp4", ".wav")

        video_path = os.path.join(video_folder, video_file)
        audio_path = os.path.join(audio_folder, audio_file)

        # Check if the corresponding audio file exists
        if os.path.exists(audio_path):
            print(f"Processing video: {video_file} with audio: {audio_file}")

            # Load video and audio
            video_clip = mp.VideoFileClip(video_path)
            audio_clip = mp.AudioFileClip(audio_path)

            # Combine video and audio
            final_clip = video_clip.set_audio(audio_clip)

            # Save the output with the original "M" prefix
            final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")
        else:
            print(f"Audio file not found for video: {video_file}")
