import torchaudio
import torch
import os
import numpy as np
from os.path import join
import argparse
from tqdm import tqdm
from scipy import interpolate

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parsing
parser = argparse.ArgumentParser('exp_motion command line tools')
parser.add_argument('--motion_folder', type=str, default="/scr/juze/datasets/BEAT2/beat_english_v2.0.0/smplxflame_30/", help="Path to the folder containing motion files")
parser.add_argument('--output_dir', type=str, default="/scr/juze/datasets/BEAT2/beat_english_v2.0.0/smplxflame_25",
                    help="Directory to save the quantized outputs")
args = parser.parse_args()

motion_folder = args.motion_folder
output_dir = args.output_dir

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each motion file in the provided folder
for motion_file in tqdm(os.listdir(motion_folder)):
    if motion_file.endswith(".npz"):
        motion_path = join(motion_folder, motion_file)
        data = np.load(motion_path)

        # # Print original shapes
        # print(f"\nFile: {motion_file}")
        # print("Original shapes:")
        # for key in data.keys():
        #     array = data[key]
        #     print(f"- {key}: {array.shape}")

        # Get the motion data
        poses = data['poses']  # (n_frames, 165)
        expressions = data['expressions']  # (n_frames, 100)
        trans = data['trans']  # (n_frames, 3)
        betas = data['betas']  # (300,)
        
        # Calculate time points
        n_frames_original = poses.shape[0]
        original_fps = 30
        target_fps = 25
        
        # Calculate exact duration and target frames
        duration = (n_frames_original - 1) / original_fps  # Total duration in seconds
        n_frames_target = int(np.floor(duration * target_fps)) + 1
        
        # Create precise time arrays
        t_original = np.linspace(0, duration, n_frames_original)
        t_target = np.linspace(0, duration, n_frames_target)
        
        # More efficient interpolation using vectorization
        def resample_sequence(sequence):
            return interpolate.interp1d(t_original, sequence, axis=0, kind='cubic', bounds_error=False, fill_value="extrapolate")(t_target)
        
        # Interpolate all motion components at once
        poses_resampled = resample_sequence(poses)
        expressions_resampled = resample_sequence(expressions)
        trans_resampled = resample_sequence(trans)
        
        # Save resampled data
        output_path = join(output_dir, motion_file)
        np.savez(output_path,
                 poses=poses_resampled,
                 expressions=expressions_resampled,
                 trans=trans_resampled,
                 betas=betas,
                 model=data['model'],
                 gender=data['gender'],
                 mocap_frame_rate=np.array(25))
        
        # # Print new shapes
        # print("\nResampled shapes:")
        # print(f"- poses: {poses_resampled.shape}")
        # print(f"- expressions: {expressions_resampled.shape}")
        # print(f"- trans: {trans_resampled.shape}")

print("Processing complete!")