import os
import moviepy.editor as mp
from collections import defaultdict
import csv
import pathlib
# Define paths
video_folder = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2/mesh_npy/"
audio_folder = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2_original/beat2_audio/"
output_folder = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2/output_videos_with_audio/mesh_npy/"

old_id_to_new_ids = defaultdict(list)
new_id_to_old_id = defaultdict(list)
for row in csv.DictReader(open('/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2/pose_data_to_joints_map.txt')):
    old_id = pathlib.Path(row['file']).with_suffix('').name
    new_id = row['id'].split('_')[0]
    old_id_to_new_ids[old_id].append(row['id'])
    old_id_to_new_ids[old_id].append('M'+row['id'])
    if new_id in new_id_to_old_id:
        continue
    new_id_to_old_id[new_id] = old_id



# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process video files
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        # Remove "M" prefix and "_mesh" suffix from video filename for matching
        # cleaned_video_name = video_file.lstrip("M").replace("_mesh", "")
        cleaned_video_name = video_file.lstrip("M").replace("_out", "").replace("_mesh", "")

        # Construct corresponding audio filename
        cleaned_video_name_old = cleaned_video_name.replace(".mp4", "")
        cleaned_video_name_new = new_id_to_old_id[cleaned_video_name_old]
        audio_file = cleaned_video_name_new + '.wav'

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
            output_file = os.path.join(output_folder, video_file)
            final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")
        else:
            print(f"Audio file not found for video: {video_file}")