import json
import os
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from lom.config import parse_args
from lom.models.build_model import build_model
from lom.utils.logger import create_logger
from os.path import join
import torchaudio
from torch import Tensor
from lom.utils.rotation_conversions import rotation_6d_to_matrix, rotation_6d_to_axis_angle, matrix_to_axis_angle, matrix_to_rotation_6d, axis_angle_to_6d
from lom.utils.other_tools import velocity2position
import math
from fairseq import checkpoint_utils
from lom.data.mixed_dataset.data_tools import (
    JOINT_MASK_UPPER,
    JOINT_MASK_HAND,
    JOINT_MASK_LOWER,
)
import smplx
import subprocess
from lom.utils.load_checkpoint import load_pretrained_vae_compositional, load_pretrained_lm
from moviepy.editor import VideoFileClip, AudioFileClip

def audio_token_to_string(audio_token: Tensor):
    audio_token = audio_token.cpu() if audio_token.device.type == 'cuda' else audio_token
    audio_list = audio_token.tolist()
    audio_string = f'<audio_id_500>'
    for j in range(len(audio_list)):
        audio_string += ''.join(f'<audio_id_{int(audio_list[j])}>')
    audio_string += f'<audio_id_501>'
    return audio_string

def smooth_moving_average(x: torch.Tensor, k: int = 21) -> torch.Tensor:
    """
    x: [B, T, C], perform moving average along the T dimension
    k: window size (odd number is better, should be <= T)
    """
    assert x.ndim == 3, "expect [B, T, C]"
    k = int(k)
    if k % 2 == 0: k += 1  # use odd kernel size
    B, T, C = x.shape
    x_ch = x.permute(0, 2, 1)                  # [B, C, T], to fit conv1d
    pad = k // 2
    x_pad = F.pad(x_ch, (pad, pad), mode="replicate")
    kernel = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / k
    y = F.conv1d(x_pad, kernel, groups=1)      # [B, 1, T]
    return y.permute(0, 2, 1)                  # back to [B, T, C]

def smooth_gaussian(x: torch.Tensor, k: int = 31, sigma: float = None) -> torch.Tensor:
    """
    Gaussian smoothing along the T dimension
    k: kernel size (odd number is better)
    sigma: standard deviation; if None, default sigma = k / 6
    """
    assert x.ndim == 3, "expect [B, T, C]"
    k = int(k)
    if k % 2 == 0: k += 1
    if sigma is None:
        sigma = k / 6.0
    B, T, C = x.shape
    x_ch = x.permute(0, 2, 1)                  # [B, C, T]
    pad = k // 2
    x_pad = F.pad(x_ch, (pad, pad), mode="replicate")
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = (g / g.sum()).view(1, 1, k)            # [1,1,k]
    y = F.conv1d(x_pad, g, groups=1)           # [B, 1, T]
    return y.permute(0, 2, 1)                  # [B, T, C]

def plot_curve(x, y, title="Curve", xlabel="X", ylabel="Y", filename="curve.png"):
    """
    Plot a curve and save it as an image.

    Args:
        x (list or np.ndarray): X-axis values
        y (list or np.ndarray): Y-axis values
        title (str): Title of the plot
        xlabel (str): Label for the X-axis
        ylabel (str): Label for the Y-axis
        filename (str): Filename to save the image (extension can be .png / .jpg / .pdf, etc.)
    """
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", linestyle="-", color="b")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # Save as image
    plt.close()  # Close the figure to avoid memory usage

def upper_token_to_string(upper_token: Tensor):
    upper_token = upper_token.cpu() if upper_token.device.type == 'cuda' else upper_token
    upper_list = upper_token.tolist()
    upper_string = f'<upper_id_256>'
    for j in range(len(upper_list)):
        upper_string += ''.join(f'<upper_id_{int(upper_list[j])}>')
    upper_string += f'<upper_id_257>'
    return upper_string

def load_text_input(txt_path):
    with open(txt_path, "r") as file:
        Lines = file.readlines()
    texts = [line for line in Lines if line.strip()]

    return_dict = {
        'text': texts,
    }

    return return_dict

def load_audio(path):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(
            wav, orig_freq=sr, new_freq=16000
        )
    return wav

def inverse_selection_tensor(filtered_t, selection_array, n):
    selection_array = torch.from_numpy(selection_array).to(filtered_t.device)
    original_shape_t = torch.zeros((n, 165)).to(filtered_t.device)
    selected_indices = torch.where(selection_array == 1)[0]
    for i in range(n):
        original_shape_t[i, selected_indices] = filtered_t[i]
    return original_shape_t

def pad_tensor(tensor, target_length):
    """Pad the given tensor to the target length or truncate it to the target length."""
    current_length = tensor.size(0)

    if current_length < target_length:
        # Pad with zeros to the target length, ensuring the padding tensor is on the same device as the original tensor
        padding_length = target_length - current_length
        return torch.cat([tensor, torch.zeros(padding_length, device=tensor.device)], dim=0)
    elif current_length > target_length:
        # If the length exceeds the target length, truncate it
        return tensor[:target_length]
    return tensor

def unify_length(outputs_face, outputs_hand, outputs_lower, outputs_upper, motion_length):
    """Unify the length of all tensors in all lists to the maximum length among the four lists."""

    max_length = motion_length
    # Unify the length of each tensor in the four lists
    outputs_face = [pad_tensor(tensor, max_length) for tensor in outputs_face]
    outputs_hand = [pad_tensor(tensor, max_length) for tensor in outputs_hand]
    outputs_lower = [pad_tensor(tensor, max_length) for tensor in outputs_lower]
    outputs_upper = [pad_tensor(tensor, max_length) for tensor in outputs_upper]

    return outputs_face, outputs_hand, outputs_lower, outputs_upper
    
def load_audio_input_tokenize(audio_path, task, hubert_checkpoint, hubert_quantizer, channel_id=None):
    """
    Load audio file and tokenize it using HuBERT model and K-means quantizer.
    
    Args:
        audio_path: Path to the audio file
        task: Task name
        hubert_checkpoint: Path to the HuBERT model checkpoint
        hubert_quantizer: Path to the K-means quantizer model
        channel_id: Audio channel to use (for stereo audio)
        
    Returns:
        Dictionary containing audio tokens
    """
    # Load the audio file
    audio = load_audio(audio_path)
    audio = audio.squeeze()
    
    # Handle stereo audio
    if len(audio.shape) == 2:
        assert (
            audio.shape[0] == 2
        ), f"expected a stereo wav of shape (2,x), found {audio.shape}"
        if channel_id is None:
            print(
                "Found stereo audio input, averaging audio from 2 channels. If you want to extract "
                "only one channel, set channel_id to 0 or 1"
            )
            audio = audio.mean(0)
        else:
            audio = audio[channel_id]
    assert len(audio.shape) == 1, audio.shape
    
    # Load models if not already provided as objects
    if isinstance(hubert_checkpoint, str):
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([hubert_checkpoint])
        hubert_model = models[0].eval().to(audio.device)
    else:
        hubert_model = hubert_checkpoint[0].eval().to(audio.device)

    # Load K-means model
    if isinstance(hubert_quantizer, str):
        import joblib
        kmeans_model = joblib.load(hubert_quantizer)
    else:
        kmeans_model = hubert_quantizer
    
    max_wav_chunk = 100 * 16_000  # 100 seconds at 16kHz
    min_wav_chunk = 400  # Minimum audio chunk size
    
    hubert_units = []
    
    # Process audio in chunks to avoid memory issues
    for start in range(0, len(audio), max_wav_chunk):
        audio_chunk = audio[start:start + max_wav_chunk]
        if len(audio_chunk) < min_wav_chunk:
            continue
            
        # Extract features using HuBERT
        audio_chunk = audio_chunk.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = hubert_model.extract_features(source=audio_chunk, padding_mask=None)[0]
        
        # Quantize features using K-means
        features = features.squeeze().cpu().numpy()
        if len(features.shape) < 2:
            features = features.reshape(1, -1)
        quantized_indices = kmeans_model.predict(features)
        
        # Convert to int and append to results
        hubert_units.extend([int(i) for i in quantized_indices])
    
    audio_token = torch.tensor(hubert_units, device=audio.device)
    
    return_dict = {
        'audio_token': audio_token,
    }
    return return_dict

def convert_smplx_to_obj_and_render(rec_pose, rec_exps, rec_trans, rec_beta, mesh_save_path, device, smplx_path, audio_path=None):
    print(f"Converting smplx to obj and rendering...(it will take a while)")

    smplx_model = smplx.create(smplx_path,
        model_type='smplx',
        gender='NEUTRAL_2020',
        use_face_contour=False,
        num_betas=300,
        num_expression_coeffs=100,
        ext='npz',
        use_pca=False,
        ).eval().to(device)
    n = rec_pose.shape[0]
    rec_pose = rec_pose.to(device)
    rec_trans = rec_trans.to(device)
    rec_exps = rec_exps.to(device)
    rec_beta = torch.tile(rec_beta, (n, 1))
    rec_beta = rec_beta.to(device)
    vertices_rec =smplx_model(
        betas=rec_beta.reshape(n, 300),
        transl=rec_trans.reshape(n, 3),
        expression=rec_exps.reshape(n, 100),
        jaw_pose=rec_pose[:, 66:69],
        global_orient=rec_pose[:, :3],
        body_pose=rec_pose[:, 3:21 * 3 + 3],
        left_hand_pose=rec_pose[:, 25 * 3:40 * 3],
        right_hand_pose=rec_pose[:, 40 * 3:55 * 3],
        leye_pose=rec_pose[:, 69:72],
        reye_pose=rec_pose[:, 72:75],
        )
    vertex_saved = vertices_rec.vertices.cpu().numpy()
    vertex_saved[:,:,1] *= -1
    np.save(mesh_save_path, vertex_saved)
    mesh_dir = os.path.dirname(mesh_save_path)
    cmd = [
        "./third_party/blender-2.93.18-linux-x64/blender",
        "--background",
        "--python", "render.py",
        "--",
        "--cfg=./configs/render.yaml",
        "--dir=" + mesh_dir,
        "--mode=video"
    ]
    # Execute command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error executing Blender: {result.stderr}")
    else:
        print("Blender rendering completed successfully")
        print(f"Output: {result.stdout}")
    
    # Merge video with audio if audio_path is provided
    if audio_path and os.path.exists(audio_path):
        print(f"Merging video with audio from: {audio_path}")
        
        try:
            # Find the generated video file (assuming it's an mp4 file in the mesh_dir)
            video_files = [f for f in os.listdir(mesh_dir) if f.endswith('.mp4')]
            
            if not video_files:
                print("Error: No video file found after rendering")
                return
            
            # Use the first video file found
            video_file = os.path.join(mesh_dir, video_files[0])
            
            # Create output filename for the video with audio
            base_name = os.path.splitext(video_files[0])[0]
            output_video_with_audio = os.path.join(mesh_dir, f"{base_name}_with_audio.mp4")
            
            # Load video and audio clips using moviepy
            video_clip = VideoFileClip(video_file)
            audio_clip = AudioFileClip(audio_path)
            
            # Get the duration of both video and audio
            video_duration = video_clip.duration
            audio_duration = audio_clip.duration
            
            print(f"Video duration: {video_duration:.2f}s, Audio duration: {audio_duration:.2f}s")
            
            # Handle duration mismatch
            if audio_duration > video_duration:
                # If audio is longer, trim it to match video duration
                audio_clip = audio_clip.subclip(0, video_duration)
                print(f"Audio trimmed to match video duration: {video_duration:.2f}s")
            elif video_duration > audio_duration:
                # If video is longer, trim it to match audio duration or loop audio
                # Option 1: Trim video to match audio
                # video_clip = video_clip.subclip(0, audio_duration)
                # print(f"Video trimmed to match audio duration: {audio_duration:.2f}s")
                
                # Option 2: Loop audio to match video duration (uncomment if preferred)
                loops_needed = int(np.ceil(video_duration / audio_duration))
                audio_clip = audio_clip.loop(duration=video_duration)
                print(f"Audio looped to match video duration: {video_duration:.2f}s")
            
            # Set the audio to the video clip
            final_video = video_clip.set_audio(audio_clip)
            
            # Write the final video with audio
            final_video.write_videofile(
                output_video_with_audio,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None  # Suppress moviepy logs
            )
            
            print(f"Successfully merged video with audio: {output_video_with_audio}")
            
            # Clean up - close the clips to free memory
            video_clip.close()
            audio_clip.close()
            final_video.close()
            
            # Optionally, remove the original video without audio
            # os.remove(video_file)
            # print(f"Removed original video file: {video_file}")
            
        except Exception as e:
            print(f"Error merging video with audio using moviepy: {e}")
            # If moviepy fails, you could fallback to the original method or handle the error
            
    else:
        if audio_path:
            print(f"Warning: Audio file not found at {audio_path}")
        else:
            print("No audio path provided, skipping audio merge")

def main():
    # parse options
    cfg = parse_args(phase="demo")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER

    # create logger
    logger = create_logger(cfg, phase="test")

    task = cfg.DEMO.TASK
    audio_path = cfg.DEMO.AUDIO
    text = cfg.DEMO.TEXT
    upper_path = cfg.DEMO.UPPER_PATH
    render = cfg.DEMO.RENDER
    motion_fps = cfg.model.params.modality_setup.params.motion_fps
    audio_fps = cfg.model.params.modality_setup.params.audio_fps

    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.target.split('.')[-2]), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(OmegaConf.to_yaml(cfg))

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")

    # create model
    total_time = time.time()
    model = build_model(cfg)
    logger.info("model {} loaded".format(cfg.model.target))

    # loading state dict
    if cfg.TEST.CHECKPOINTS:
        logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
        # load_pretrained_vae(cfg, model, logger, phase="demo")
        load_pretrained_vae_compositional(cfg, model, logger, phase="demo")
        load_pretrained_lm(cfg, model, logger, phase="demo")
    else:
        logger.warning(
            "No checkpoints provided, using random initialized model")

    model.to(device)
    model.to(torch.bfloat16)

    motion_token_fps = cfg.model.params.lm.params.motion_framerate / cfg.model.params.lm.params.motion_down_sampling
    upper_root_dir = '/scr/juze/datasets/HumanML3D_25/TOKENS_AGENT_25_Rotation/upper'
    save_reconstruced = True

    upper_paths = os.listdir(upper_root_dir)

    for upper_name in tqdm(upper_paths, desc="Processing upper paths"):
        save_path_hand = os.path.join(upper_root_dir.replace('upper', 'hand_generated'), upper_name)
        if os.path.exists(save_path_hand):
            continue

        # audio_short_length = int(cfg.TEST.TEST_LENGTH * audio_token_fps / motion_token_fps)        
        upper_token = np.load(os.path.join(upper_root_dir, upper_name))
        upper_token = torch.from_numpy(upper_token)
        upper_token = upper_token.to(device)
        pose_token_length = upper_token.shape[1]
        upper_short_length = int(cfg.TEST.TEST_LENGTH)     
        num_subdivision = math.floor(pose_token_length / upper_short_length) + 1

        # Initialize containers for token indices
        rec_index_all_face = []
        rec_index_all_upper = []
        rec_index_all_lower = []
        rec_index_all_hands = []
        # num_subdivision = 1
        for index in range(num_subdivision):
            # Calculate audio chunk indices
            upper_start = math.floor(index * upper_short_length)
            upper_end = upper_start + upper_short_length

            upper_tokens_tmp = upper_token[0, upper_start:upper_end]
            upper_token_string = upper_token_to_string(upper_tokens_tmp)
            batch = {
                "upper":
                [upper_token_string]
            }
            outputs = model(batch, task=task)
            # Extract results - these are already tokens from the encoder-decoder generation
            outputs_face = outputs['face']
            outputs_hand = outputs['hand']
            outputs_lower = outputs['lower']
            outputs_upper = outputs['upper']
            outputs_upper = upper_tokens_tmp.unsqueeze(0)
            ## convert to float, output is list of tensors
            outputs_face = [tensor.float() for tensor in outputs_face]
            outputs_hand = [tensor.float() for tensor in outputs_hand]
            outputs_lower = [tensor.float() for tensor in outputs_lower]
            outputs_upper = [tensor.float() for tensor in outputs_upper]

            # Unify tensor lengths
            feats_face, feats_hand, feats_lower, feats_upper = unify_length(
                outputs_face, outputs_hand, outputs_lower, outputs_upper, 
                cfg.TEST.TEST_LENGTH)    

            # Stack tensors
            feats_face = torch.stack(feats_face, dim=0)
            feats_hand = torch.stack(feats_hand, dim=0)
            feats_lower = torch.stack(feats_lower, dim=0)
            feats_upper = torch.stack(feats_upper, dim=0)

            rec_index_all_face.append(feats_face)
            rec_index_all_upper.append(feats_upper)
            rec_index_all_lower.append(feats_lower)
            rec_index_all_hands.append(feats_hand)

        # Concatenate all chunks
        rec_index_face = torch.cat(rec_index_all_face, dim=1)
        rec_index_upper = torch.cat(rec_index_all_upper, dim=1)
        rec_index_lower = torch.cat(rec_index_all_lower, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)

        rec_index_hands = rec_index_hands[:, :pose_token_length]
        rec_index_upper = rec_index_upper[:, :pose_token_length]
        rec_index_lower = rec_index_lower[:, :pose_token_length]
        rec_index_face = rec_index_face[:, :pose_token_length]

        # Ensure token indices are valid for the vocabulary size
        rec_index_face = torch.clamp(rec_index_face, 0, model.lm.face_codebook_size - 1)
        rec_index_upper = torch.clamp(rec_index_upper, 0, model.lm.upper_codebook_size - 1)
        rec_index_lower = torch.clamp(rec_index_lower, 0, model.lm.lower_codebook_size - 1)
        rec_index_hands = torch.clamp(rec_index_hands, 0, model.lm.hand_codebook_size - 1)

        rec_index_hands = rec_index_hands.cpu().numpy().astype(np.int64)
        np.save(save_path_hand, rec_index_hands)



if __name__ == "__main__":
    main()
