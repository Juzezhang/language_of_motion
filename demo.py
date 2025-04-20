import json
import os
from pathlib import Path
import time
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf
from tqdm import tqdm
from lom.config import parse_args
from lom.data.build_data import build_data
from lom.models.build_model import build_model
from lom.utils.logger import create_logger
import lom.render.matplot.plot_3d_global as plot_3d
from os.path import join
# from lom.archs.hubert.hubert_tokenizer import HubertTokenizer
import torchaudio
from torch import Tensor
from typing import List, Union, Dict, Any
from lom.utils.rotation_conversions import rotation_6d_to_matrix, rotation_6d_to_axis_angle, matrix_to_axis_angle, matrix_to_rotation_6d, axis_angle_to_6d
from lom.utils.other_tools import velocity2position, estimate_linear_velocity
import math
from fairseq import checkpoint_utils
from lom.data.mixed_dataset.data_tools import (
    JOINT_MASK_UPPER,
    JOINT_MASK_HANDS,
    JOINT_MASK_LOWER,
)
import smplx
import subprocess

def audio_token_to_string(audio_token: Tensor):
    audio_token = audio_token.cpu() if audio_token.device.type == 'cuda' else audio_token
    audio_list = audio_token.tolist()
    audio_string = f'<audio_id_500>'
    for j in range(len(audio_list)):
        audio_string += ''.join(f'<audio_id_{int(audio_list[j])}>')
    audio_string += f'<audio_id_501>'
    return audio_string

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

def convert_smplx_to_obj(rec_pose, rec_exps, rec_trans, rec_beta, mesh_save_path, device, smplx_path):

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
    # 执行命令并捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error executing Blender: {result.stderr}")
    else:
        print("Blender rendering completed successfully")
        print(f"Output: {result.stdout}")



def main():
    # parse options
    cfg = parse_args(phase="demo")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER

    # create logger
    logger = create_logger(cfg, phase="test")

    task = cfg.DEMO.TASK
    audio_path = cfg.DEMO.AUDIO
    text = cfg.DEMO.TEXT
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

    # # Dataset
    # datamodule = build_data(cfg)
    # logger.info("datasets module {} initialized".format("".join(
    #     cfg.DATASET.target.split('.')[-2])))

    # create model
    total_time = time.time()
    # model = build_model(cfg, datamodule)
    model = build_model(cfg)
    logger.info("model {} loaded".format(cfg.model.target))

    # loading state dict
    if cfg.TEST.CHECKPOINTS:
        logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
        state_dict = torch.load(cfg.TEST.CHECKPOINTS, weights_only=False
                                map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
    else:
        logger.warning(
            "No checkpoints provided, using random initialized model")

    model.to(device)
    model.to(torch.bfloat16)

    if cfg.DEMO.AUDIO and task == 'a2m':

        # audio_tokenizer_path = cfg.TRAIN.AUDIO_TOKENIZER

        hubert_checkpoint = cfg.TRAIN.HUBERT_CHECKPOINT
        hubert_quantizer = cfg.TRAIN.HUBERT_QUANTIZER

        audio_tokenizer, _, task = checkpoint_utils.load_model_ensemble_and_task([hubert_checkpoint])
        # load audio input
        return_dict = load_audio_input_tokenize(audio_path, task, audio_tokenizer, hubert_quantizer)
        audio_token = return_dict['audio_token'].to(device)

        # audio_token = np.load('/scr/juze/datasets/BEAT2/beat_english_v2.0.0/audios_token/2_scott_0_111_111.npy')
        audio_token_length = audio_token.shape[0]
        # Calculate number of chunks and their sizes
        audio_token_fps = cfg.DATASET.audio_fps / cfg.DATASET.audio_down
        motion_token_fps = cfg.DATASET.pose_fps / cfg.DATASET.unit_length

        audio_short_length = int(cfg.TEST.TEST_LENGTH * audio_token_fps / motion_token_fps)        
        num_subdivision = math.floor(audio_token_length / audio_short_length) + 1

        # Initialize containers for token indices
        rec_index_all_face = []
        rec_index_all_upper = []
        rec_index_all_lower = []
        rec_index_all_hands = []

        for index in tqdm(range(num_subdivision), desc="Processing audio chunks"):
            # Calculate audio chunk indices
            audio_start = math.floor(index * audio_short_length)
            audio_end = audio_start + audio_short_length

            audio_tokens_tmp = audio_token[audio_start:audio_end]
            audio_token_string = audio_token_to_string(audio_tokens_tmp)
            batch = {
                "audio":
                [audio_token_string]
            }
            outputs = model(batch, task=task)
            # Extract results - these are already tokens from the encoder-decoder generation
            outputs_face = outputs['face']
            outputs_hand = outputs['hand']
            outputs_lower = outputs['lower']
            outputs_upper = outputs['upper']

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

        # Ensure token indices are valid for the vocabulary size
        rec_index_face = torch.clamp(rec_index_face, 0, model.lm.face_codebook_size - 1)
        rec_index_upper = torch.clamp(rec_index_upper, 0, model.lm.upper_codebook_size - 1)
        rec_index_lower = torch.clamp(rec_index_lower, 0, model.lm.lower_codebook_size - 1)
        rec_index_hands = torch.clamp(rec_index_hands, 0, model.lm.hand_codebook_size - 1)

    elif text and task == 't2m':
        rec_index_all_face = []
        rec_index_all_upper = []
        rec_index_all_lower = []
        rec_index_all_hands = []
        # load text input
        return_dict = load_text_input(text)
    
        batch = {
            "text":
            return_dict['text']
        }
        outputs = model(batch, task=task)
        # Extract results - these are already tokens from the encoder-decoder generation
        outputs_face = outputs['face']
        outputs_hand = outputs['hand']
        outputs_lower = outputs['lower']
        outputs_upper = outputs['upper']

        ## convert to float, output is list of tensors
        outputs_face = [tensor.float() for tensor in outputs_face]
        outputs_hand = [tensor.float() for tensor in outputs_hand]
        outputs_lower = [tensor.float() for tensor in outputs_lower]
        outputs_upper = [tensor.float() for tensor in outputs_upper]

        # Unify tensor lengths
        feats_face, feats_hand, feats_lower, feats_upper = unify_length(
            outputs_face, outputs_hand, outputs_lower, outputs_upper, 
            min(outputs_face[0].shape[0], outputs_hand[0].shape[0], outputs_lower[0].shape[0], outputs_upper[0].shape[0]))

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

        # Ensure token indices are valid for the vocabulary size
        rec_index_face = torch.clamp(rec_index_face, 0, model.lm.face_codebook_size - 1)
        rec_index_upper = torch.clamp(rec_index_upper, 0, model.lm.upper_codebook_size - 1)
        rec_index_lower = torch.clamp(rec_index_lower, 0, model.lm.lower_codebook_size - 1)
        rec_index_hands = torch.clamp(rec_index_hands, 0, model.lm.hand_codebook_size - 1)

    model.vae_global.float()
    model.vae_face.float()
    model.vae_upper.float()
    model.vae_lower.float()
    model.vae_hand.float()

    # Decode tokens to motion representations using motion tokenizers (VAEs)
    rec_face = model.vae_face.decode(rec_index_face.int()).float()
    rec_upper = model.vae_upper.decode(rec_index_upper.int()).float()
    rec_lower = model.vae_lower.decode(rec_index_lower.int()).float()
    rec_hands = model.vae_hand.decode(rec_index_hands.int()).float()

    # Preparation for pose reconstruction
    rec_face = rec_face.float()
    rec_exps = rec_face[:, :, 6:]
    rec_pose_jaw = rec_face[:, :, :6]
    rec_pose_legs = rec_lower[:, :, :54]
    bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]
    
    # Convert 6D rotations to axis-angle format for different body parts
    # Upper body
    rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
    rec_pose_upper = rotation_6d_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)
    rec_pose_upper_recover = inverse_selection_tensor(rec_pose_upper.cuda(), JOINT_MASK_UPPER, bs*n)
    
    # Lower body
    rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
    rec_pose_lower = rotation_6d_to_matrix(rec_pose_lower)
    rec_lower2global = matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9 * 6)
    rec_pose_lower = matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
    rec_pose_lower_recover = inverse_selection_tensor(rec_pose_lower.cuda(), JOINT_MASK_LOWER, bs * n)
    
    # Hands
    rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
    rec_pose_hands = rotation_6d_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)
    rec_pose_hands_recover = inverse_selection_tensor(rec_pose_hands.cuda(), JOINT_MASK_HANDS, bs * n)
    
    # Jaw
    rec_pose_jaw = rec_pose_jaw.reshape(bs * n, 6)
    rec_pose_jaw = rotation_6d_to_axis_angle(rec_pose_jaw).reshape(bs * n, 1 * 3)
    
    # Combine all body parts into a full-body pose
    rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
    rec_pose[:, 66:69] = rec_pose_jaw

    # Process global motion (translation, root orientation)
    to_global = rec_lower
    if to_global.shape[2] == 54:
        to_global = F.pad(to_global, (0, 7))
    to_global[:, :, 54:57] = 0.0
    to_global[:, :, :54] = rec_lower2global
    rec_global = model.vae_global(to_global)

    # Calculate translation from velocities
    rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
    rec_x_trans = velocity2position(rec_trans_v_s[:, :, 0:1], 1 / motion_fps, torch.zeros(rec_trans_v_s[:, 0, 0:1].shape, device=device))
    rec_z_trans = velocity2position(rec_trans_v_s[:, :, 2:3], 1 / motion_fps, torch.zeros(rec_trans_v_s[:, 0, 2:3].shape, device=device))
    rec_y_trans = rec_trans_v_s[:, :, 1:2]
    rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
    
    rec_beta = torch.zeros(300)
    rec_beta_np = rec_beta.numpy()
    rec_pose_np = rec_pose.cpu().numpy().reshape(n, 55 * 3)
    rec_exp_np = rec_exps.cpu().numpy().reshape(n, 100)
    rec_trans_np = rec_trans.cpu().numpy().reshape(n, 3)

    if audio_path:
        save_name = os.path.basename(audio_path).replace('.wav', '.npz')
        np.savez( os.path.join(output_dir, save_name),
                    betas=rec_beta_np,
                    poses= rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate=30,
                )
    elif text:
        save_name = os.path.basename(text).replace('.txt', '.npz')
        np.savez( os.path.join(output_dir, save_name),
                    betas=rec_beta_np,
                    poses= rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate=30,
                )
        


    if render:
        smplx_path = cfg.RENDER.SMPLX2020_MODEL_PATH
        mesh_save_name = save_name.replace('.npz', '.npy')
        mesh_save_path = os.path.join(output_dir, mesh_save_name)
        convert_smplx_to_obj(rec_pose, rec_exps, rec_trans, rec_beta, mesh_save_path, device, smplx_path)


    # '/root/blender-2.93.18-linux-x64/blender'


    logger.info('Model forward finished! Start saving results...')

    total_time = time.time() - total_time
    logger.info(
        f'Total time spent: {total_time:.2f} seconds (including model loading time and exporting time).'
    )
    logger.info(f"Testing done, the npy are saved to {output_dir}")


if __name__ == "__main__":
    main()
