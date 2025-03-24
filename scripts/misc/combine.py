from glob import glob
import os
from tqdm import tqdm
import numpy as np
from lom.data.mixed_dataset.data_tools import joints_list

upbody = '/nas/nas_32/AI-being/zhangjz/exp_motion/paper_result/vq_abalation/upper/result/gt/'
lowbody = '/nas/nas_32/AI-being/zhangjz/exp_motion/paper_result/vq_abalation/lower/result/rec/'

up_files = sorted(
    glob(os.path.join(upbody, "*.npz")),
    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
)
low_files = sorted(
    glob(os.path.join(lowbody, "*.npz")),
    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
)
columns_to_copy = [9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38, 39, 40, 41, 42, 43,
                   44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                   61, 62, 63, 64, 65]




upbody = '/nas/nas_32/AI-being/zhangjz/exp_motion/paper_result/vq_abalation/upper/result/rec/'
lowbody = '/nas/nas_32/AI-being/zhangjz/exp_motion/paper_result/vq_abalation/lower/result/rec/'

up_files = sorted(
    glob(os.path.join(upbody, "*.npz")),
    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
)
low_files = sorted(
    glob(os.path.join(lowbody, "*.npz")),
    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
)
columns_to_copy = [9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38, 39, 40, 41, 42, 43,
                   44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                   61, 62, 63, 64, 65]



ori_joint_list = joints_list['beat_smplx_joints']
tar_joint_list_upper = joints_list['beat_smplx_upper']
tar_joint_list_lower = joints_list['beat_smplx_lower']
# del tar_joint_list_lower['pelvis']


joint_mask_upper = np.zeros(len(list(ori_joint_list.keys())) * 3)
for joint_name in tar_joint_list_upper:
    joint_mask_upper[ ori_joint_list[joint_name][1] - ori_joint_list[joint_name][0]:ori_joint_list[joint_name][1]] = 1

joint_mask_lower = np.zeros(len(list(ori_joint_list.keys())) * 3)
for joint_name in tar_joint_list_lower:
    joint_mask_lower[ ori_joint_list[joint_name][1] - ori_joint_list[joint_name][0]:ori_joint_list[joint_name][1]] = 1


save_rec_dir = '/nas/nas_32/AI-being/zhangjz/exp_motion/paper_result/vq_abalation/com/result/rec'
os.makedirs(save_rec_dir, exist_ok=True)

for idx, (up_file, low_file) in tqdm(enumerate(zip(up_files, low_files)), total=len(up_files)):
    up_data = np.load(up_file, allow_pickle=True)
    low_data = np.load(low_file, allow_pickle=True)
    gt_data = np.load(low_file.replace('rec',"gt"), allow_pickle=True)
    updated_data = {}
    for key in low_data.keys():
        if key == 'poses':
            updated_pose_data = gt_data['poses'].copy()
            n = updated_pose_data.shape[0]
            updated_pose_data[:n, joint_mask_upper.astype(bool)] = up_data['poses'][:n, joint_mask_upper.astype(bool)]
            updated_pose_data[:n, joint_mask_lower.astype(bool)] = low_data['poses'][:n, joint_mask_lower.astype(bool)]
            updated_data[key] = updated_pose_data
            if updated_data[key].shape[0] != gt_data['poses'].shape[0]:
                a= 1
        else:
            updated_data[key] = low_data[key]

    save_path = os.path.join(save_rec_dir, f"rec_{idx}.npz")
    np.savez(save_path, **updated_data)
