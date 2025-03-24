import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, clips_array, TextClip, CompositeVideoClip

# Define paths for the input video folders
video_path1 = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/AMASS/mesh_save/"
video_path2 = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/AMASS/reconstructed_motion_ds4/"

# Define output path for the merged videos
output_path = "/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/AMASS/reconstructed_compare_ds4/"
os.makedirs(output_path, exist_ok=True)

# Get all .mp4 files in each folder
files_in_folder1 = set([f for f in os.listdir(video_path1) if f.endswith('.mp4')])
files_in_folder2 = set([f for f in os.listdir(video_path2) if f.endswith('.mp4')])

# Find common files that exist in both folders
common_files = files_in_folder1.intersection(files_in_folder2)

# Process and merge each common file
for file_name in common_files:
    # Build full paths for each video file
    video_file1 = os.path.join(video_path1, file_name)
    video_file2 = os.path.join(video_path2, file_name)

    # Load video clips
    video_clip1 = VideoFileClip(video_file1)
    video_clip2 = VideoFileClip(video_file2)

    # Ensure both videos have the same duration by setting to the shorter duration
    min_duration = min(video_clip1.duration, video_clip2.duration)
    video_clip1 = video_clip1.subclip(0, min_duration)
    video_clip2 = video_clip2.subclip(0, min_duration)

    # Add text annotations to each video
    txt_clip1 = TextClip("GroundTruth", fontsize=30, color='black', font='Arial').set_position(
        ("center", "bottom")).set_duration(min_duration)
    txt_clip2 = TextClip("Reconstructed Motion", fontsize=30, color='black', font='Arial').set_position(
        ("center", "bottom")).set_duration(min_duration)

    # Overlay the text annotations on each video
    video_with_txt1 = CompositeVideoClip([video_clip1, txt_clip1])
    video_with_txt2 = CompositeVideoClip([video_clip2, txt_clip2])

    # Combine videos side-by-side
    final_video = clips_array([[video_with_txt1, video_with_txt2]])

    # Define the output file path for the merged video
    output_file = os.path.join(output_path, file_name)

    # Export the final merged video with specified codec settings
    final_video.write_videofile(output_file, codec="libx264", audio_codec="aac")

    # Release resources after processing each video
    video_clip1.close()
    video_clip2.close()

print(f"Finished. The output path is: {output_path}")