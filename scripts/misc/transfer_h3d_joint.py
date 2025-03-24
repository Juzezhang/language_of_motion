import os
import random
import torch
import numpy as np
import imageio
import moviepy.editor as mp
from pathlib import Path
from scipy.spatial.transform import Rotation as RRR
import lom.render.matplot.plot_3d_global as plot_3d
from lom.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from lom.data.humanml.scripts.motion_process import process_file, recover_from_ric
from lom.config import parse_args
from os.path import join

# Set environment variables for rendering
os.environ['DISPLAY'] = ':0.0'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Set device for computation
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Quaternion Operations ---
def qinv(q):
    """ Compute the inverse of a quaternion. """
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qinv_np(q):
    """ Inverse quaternion using NumPy. """
    return qinv(torch.from_numpy(q).float()).numpy()


def qrot(q, v):
    """
    Rotate vector(s) v using quaternion(s) q.
    Expects tensor q of shape (*, 4) and vector v of shape (*, 3).
    Returns rotated vector with shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qrot_np(q, v):
    """ Rotate vector using quaternion with NumPy inputs. """
    q = torch.from_numpy(q).float()
    v = torch.from_numpy(v).float()
    return qrot(q, v).numpy()


# --- Transformation ---
def rigid_transform_joint(relative, data):
    """
    Apply rigid transformation to joint positions.
    Args:
        relative: Relative transformation (rotation + translation).
        data: Joint positions with shape (seq_len, 22, 3).
    Returns:
        Transformed joint positions.
    """
    data = data.reshape(-1, 22 * 3)
    global_positions = data[..., :22 * 3].reshape(data.shape[:-1] + (22, 3))

    relative_rot = relative[0]
    relative_t = relative[1:3]

    relative_r_rot_quat = np.zeros(global_positions.shape[:-1] + (4,))
    relative_r_rot_quat[..., 0] = np.cos(relative_rot)
    relative_r_rot_quat[..., 2] = np.sin(relative_rot)

    global_positions = qrot_np(qinv_np(relative_r_rot_quat), global_positions)
    global_positions[..., [0, 2]] += relative_t

    data[..., :22 * 3] = global_positions.reshape(data.shape[:-1] + (-1,))
    return data.reshape(-1, 22, 3)


# --- Process Pose ---
# def process_pose(pose, relative=None):
#     """
#     Process pose data for two characters and apply relative transformation.
#     """
#
#
#     motion1 = pose[..., :263][0]
#     motion2 = pose[..., 263:526][0]
#     motion2 = np.zeros_like(pose[..., :263])[0]
#
#     if relative is None:
#         relative = pose[..., 526:][0, 0]
#         relative = np.zeros(3)
#     rec_motion1 = recover_from_ric(torch.from_numpy(motion1).unsqueeze(0).float(), 22)[0].numpy()
#     rec_motion2 = recover_from_ric(torch.from_numpy(motion2).unsqueeze(0).float(), 22)[0].numpy()
#
#     rec_motion2 = rigid_transform_joint(relative, rec_motion2)
#     return np.stack([rec_motion1, rec_motion2], axis=1), relative

def process_pose(pose, joints_num=22):
    """
    Process pose data for an arbitrary number of characters with relative transformations.

    Args:
        pose (numpy array): Input pose array with shape (n_samples, n_frames, total_dim),
                            where total_dim = 263 + 3 + 263 + 3 + ... for n characters.
        joints_num (int): Number of joints in the pose data.

    Returns:
        numpy array: Processed pose array of shape (n_samples, n_frames, n_characters, joints_num, 3).
    """
    num_motions = (pose.shape[-1] + 3) // (263 + 3)  # Calculate number of characters
    processed_motions = []
    current_relative = np.zeros(3)  # Initial relative transformation
    motion_data = pose[..., :263][0]
    # Recover joint positions
    rec_motion = recover_from_ric(torch.from_numpy(motion_data).unsqueeze(0).float(), joints_num)[0].numpy()
    rec_motion = rigid_transform_joint(current_relative, rec_motion)
    processed_motions.append(rec_motion)

    if num_motions > 1:
        for i in range(1, num_motions):
            # Extract motion data
            motion_start = (i-1) * 266 + 263
            motion_end = motion_start + 263
            motion_data = pose[..., motion_start:motion_end][0]

            # Recover joint positions
            rec_motion = recover_from_ric(torch.from_numpy(motion_data).unsqueeze(0).float(), joints_num)[0].numpy()

            # Update current relative transformation for the next character
            current_relative = pose[..., motion_end:motion_end + 3][0,0]

            # Apply relative transformation if not the first character
            rec_motion = rigid_transform_joint(current_relative, rec_motion)

            # Append processed motion
            processed_motions.append(rec_motion)


    return np.stack(processed_motions, axis=1)




# --- Render Motion ---
def render_motion(file_name, pose_data, feature_data, output_dir):
    """
    Render a motion sequence and save it as a video.
    """
    video_file = file_name + '.mp4'
    feature_file = file_name + '.npy'

    output_feature_path = os.path.join(output_dir, feature_file)
    output_video_path = os.path.join(output_dir, video_file)

    np.save(output_feature_path, feature_data)

    gif_path = output_video_path[:-4] + '.gif'

    if len(pose_data.shape) == 4:
        pose_data = pose_data[None]

    if isinstance(pose_data, torch.Tensor):
        pose_data = pose_data.cpu().numpy()

    pose_vis = plot_3d.draw_to_batch(pose_data, [''], [gif_path])
    out_video = mp.VideoFileClip(gif_path)
    out_video.write_videofile(output_video_path)

    del pose_vis
    return output_video_path, video_file, output_feature_path, feature_file


# --- Main Execution ---
if __name__ == "__main__":
    # Load configuration
    cfg = parse_args(phase="webui")

    # Directories for saving joint and visualization outputs
    save_joint_dir = '/nas/nas_32/AI-being/zhangjz/social_motion/experiments/groundtruth_joint_selected/'
    os.makedirs(save_joint_dir, exist_ok=True)

    # Process pose data
    # data_dir = '/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/humanml3d_heng'
    # pose_dir = join(data_dir, 'new_joint_vecs')
    pose_dir = '/nas/nas_32/AI-being/zhangjz/social_motion/experiments/groundtruth_selected/'

    # sample_test = np.load()

    for file in sorted(os.listdir(pose_dir)):
        if file.endswith('.npy'):
            pose = np.load(os.path.join(pose_dir, file),allow_pickle=True)[None, ...]
            processed_pose = process_pose(pose)

            save_path = join(save_joint_dir, file)
            np.save(save_path, processed_pose)