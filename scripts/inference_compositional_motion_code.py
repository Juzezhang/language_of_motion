import os
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from tqdm import tqdm
from lom.config import parse_args# from lom.models.build_model import build_model
from loguru import logger
# from lom.data.beat2.data_tools import joints_list
import pandas as pd
from os.path import join as pjoin
import codecs as cs
import torch.nn.functional as F
from lom.utils.load_checkpoint import load_pretrained_vae_compositional
from lom.models.build_model import build_model
import random
from lom.data.mixed_dataset.data_tools import (
    joints_list, 
    JOINT_MASK_FACE,
    JOINT_MASK_UPPER,
    JOINT_MASK_HAND,
    JOINT_MASK_LOWER,
    JOINT_MASK_FULL,
    BEAT_SMPLX_JOINTS,
    BEAT_SMPLX_FULL,
    BEAT_SMPLX_FACE,
    BEAT_SMPLX_UPPER,
    BEAT_SMPLX_HAND,
    BEAT_SMPLX_LOWER
)
from lom.utils.rotation_conversions import axis_angle_to_6d, axis_angle_to_matrix, rotation_6d_to_axis_angle, axis_angle_to_6d_np
from lom.utils.rotation_conversions import rotation_6d_to_matrix, rotation_6d_to_axis_angle, matrix_to_axis_angle, matrix_to_rotation_6d, axis_angle_to_matrix

from lom.utils.other_tools import velocity2position, estimate_linear_velocity
import smplx
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip



def inverse_selection_tensor(filtered_t, selection_array, n):
    selection_array = torch.from_numpy(selection_array).cuda()
    original_shape_t = torch.zeros((n, 165)).cuda()
    selected_indices = torch.where(selection_array == 1)[0]
    for i in range(n):
        original_shape_t[i, selected_indices] = filtered_t[i]
    return original_shape_t



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
    with torch.no_grad():
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
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1

    Random_is = True  ## Set a parameter to control if we want to randomly sample a pose from the dataset
    render = False

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)
    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load each dataset based on its type from the configuration
    path_name_list = []
    dataset_name_list = []

    for config in cfg.DATASET.datasets:
        dataset_name = config.get("name")
        if dataset_name == "AMASS":
            data_root = cfg.DATASET['AMASS'].ROOT
            output_dir_amass = os.path.join(data_root, 'reconstructed_motion_ds4_25')
            os.makedirs(output_dir_amass, exist_ok=True)
            split_file_test = pjoin(data_root, 'test.txt')
            amass_root = data_root
            # Data id list
            id_list_test = []
            with cs.open(split_file_test, "r") as f:
                for line in f.readlines():
                    id_list_test.append(line.strip())
            if Random_is:
                id_list_test = random.sample(id_list_test, 1)
            id_list_test = ['M013765']
            for index, file_name in tqdm(enumerate(id_list_test)):
                ext = ".npy"
                path_name_list.append( file_name + ext)
                dataset_name_list.append('amass')
        if dataset_name == "BEAT2":
            data_root = cfg.DATASET['BEAT2'].ROOT
            output_dir_beat2 = os.path.join(data_root, 'reconstructed_motion_ds4_25')
            os.makedirs(output_dir_beat2, exist_ok=True)
            split_rule = pd.read_csv(pjoin(data_root,"train_test_split.csv"))
            selected_file = split_rule.loc[(split_rule['type'] == 'test')]
            if Random_is:
                selected_file = selected_file.sample(1)
            beat2_root = data_root
            for index, file_name in selected_file.iterrows():
                f_name = file_name["id"]
                ext = ".npy"
                path_name_list.append(f_name + ext)
                dataset_name_list.append('beat2')


    # Model
    model = build_model(cfg)
    logger.info("model {} loaded".format(cfg.model.target))

    # load_pretrained_vae(cfg, model, logger, phase="token")
    load_pretrained_vae_compositional(cfg, model, logger, phase="token")

    if cfg.ACCELERATOR == "gpu":
        device = 'cuda'
        model.vae_face.to('cuda')
        model.vae_upper.to('cuda')
        model.vae_hand.to('cuda')
        model.vae_lower.to('cuda')
        model.vae_global.to('cuda')

    model.vae_face.eval()
    model.vae_upper.eval()
    model.vae_hand.eval()
    model.vae_lower.eval()
    model.vae_global.eval()

    logger.info("model loaded")
    
    pose_fps = cfg.DATASET.pose_fps
    j = 55

    # for batch in tqdm(datasets.token_dataloader(), desc=f'motion tokenize'):
    for data_index in tqdm(range(len(path_name_list)), desc=f'motion tokenize'):

        path_name = path_name_list[data_index]
        dataset_name = dataset_name_list[data_index]

        try:
            if dataset_name == 'amass':
                data_root = amass_root
                output_dir = output_dir_amass
                gt_npz = np.load(pjoin(data_root, 'amass_data_align_25', path_name.split('.')[0] + ".npz"), allow_pickle=True)
            else:
                data_root = beat2_root
                output_dir = output_dir_beat2
                gt_npz = np.load(pjoin(data_root, 'smplxflame_25', path_name.split('.')[0] + ".npz"), allow_pickle=True)
            hand_token = np.load(pjoin(data_root, 'TOKENS_DS4_25_PaperVersion', 'hands', path_name))
            hand_token = torch.from_numpy(hand_token).to('cuda').int()
            upper_token = np.load(pjoin(data_root, 'TOKENS_DS4_25_PaperVersion', 'upper', path_name))
            upper_token = torch.from_numpy(upper_token).to('cuda').int()
            lower_token = np.load(pjoin(data_root, 'TOKENS_DS4_25_PaperVersion', 'lower', path_name))
            lower_token = torch.from_numpy(lower_token).to('cuda').int()
        except:
            continue

        if dataset_name == 'amass':
            face_token = torch.zeros([1, lower_token.shape[1]], dtype=torch.int).to('cuda')
            rec_face = model.vae_face.decode(face_token)
        else:
            face_token = np.load(pjoin(data_root, 'TOKENS_DS4_25_PaperVersion', 'face', path_name))
            face_token = torch.from_numpy(face_token).to('cuda').int()
            rec_face = model.vae_face.decode(face_token)

        rec_upper = model.vae_upper.decode(upper_token)
        rec_lower = model.vae_lower.decode(lower_token)
        rec_hands = model.vae_hand.decode(hand_token)

        bs, n = rec_upper.shape[0], rec_upper.shape[1]

        rec_exps = rec_face[:, :, 6:]
        rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rotation_6d_to_matrix(rec_pose_upper)  #
        rec_pose_upper = matrix_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)
        rec_pose_upper_recover = inverse_selection_tensor(rec_pose_upper, JOINT_MASK_UPPER, bs * n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9 * 6)
        rec_pose_lower = matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
        rec_pose_lower_recover = inverse_selection_tensor(rec_pose_lower, JOINT_MASK_LOWER, bs * n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = matrix_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)
        rec_pose_hands_recover = inverse_selection_tensor(rec_pose_hands, JOINT_MASK_HAND, bs * n)
        rec_pose_jaw = rec_pose_jaw.reshape(bs * n, 6)
        rec_pose_jaw = rotation_6d_to_matrix(rec_pose_jaw)
        rec_pose_jaw = matrix_to_axis_angle(rec_pose_jaw).reshape(bs * n, 1 * 3)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        rec_pose[:, 66:69] = rec_pose_jaw

        to_global = rec_lower
        if to_global.shape[2] == 54:
            to_global = F.pad(to_global, (0, 7))
        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = model.vae_global(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        # rec_x_trans = velocity2position(rec_trans_v_s[:, :, 0:1].cpu(), 1 / pose_fps,
        #                                             torch.zeros(rec_trans_v_s[:, 0, 0:1].cpu().shape))
        # rec_z_trans = velocity2position(rec_trans_v_s[:, :, 2:3].cpu(), 1 / pose_fps,
        #                                             torch.zeros(rec_trans_v_s[:, 0, 0:1].cpu().shape))
        rec_x_trans = velocity2position(rec_trans_v_s[:, :, 0:1].cpu(), 1 / pose_fps, torch.tensor(gt_npz["trans"][0:1, 0:1]))
        rec_z_trans = velocity2position(rec_trans_v_s[:, :, 2:3].cpu(), 1 / pose_fps, torch.tensor(gt_npz["trans"][0:1, 2:3]))
        rec_y_trans = rec_trans_v_s[:, :, 1:2].cpu()
        rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        rec_pose = axis_angle_to_matrix(rec_pose.reshape(bs * n, j, 3))
        rec_pose = matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)

        ####  SAVE
        # print(rec_pose.shape, tar_pose.shape)
        rec_pose = rotation_6d_to_matrix(rec_pose.reshape(bs * n, j, 6))
        rec_pose = matrix_to_axis_angle(rec_pose).reshape(bs * n, j * 3)

        rec_pose_np = rec_pose.detach().cpu().numpy()
        rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs * n, 3)
        rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs * n, 100)

        tar_beta_np = gt_npz["betas"]
        tar_beta = torch.from_numpy(tar_beta_np).to('cuda')
        np.savez(pjoin(output_dir, path_name.split('.')[0] + '.npz'),
                 betas=tar_beta_np,
                 poses=rec_pose_np,
                 expressions=rec_exp_np,
                 trans=rec_trans_np,
                 model='smplx2020',
                 gender='neutral',
                 mocap_frame_rate=pose_fps,
                 )
        if render:
            smplx_path = cfg.RENDER.SMPLX2020_MODEL_PATH
            mesh_save_name = path_name.replace('.npz', '.npy')
            mesh_save_path = os.path.join(output_dir, mesh_save_name)
            convert_smplx_to_obj_and_render(rec_pose, rec_exps, rec_trans, tar_beta, mesh_save_path, device, smplx_path)
            pass


    print(
        f'Motion reconstruction done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()
