from __future__ import division, print_function
import argparse
import os
from os.path import isfile, join
import torch
import numpy as np
import joblib
import smplx
import trimesh
import h5py
from pathlib import Path
from mGPT.data.transforms.joints2rots import config
from mGPT.data.transforms.joints2rots.smplify import SMPLify3D
from mGPT.utils.joints import mmm_to_smplh_scaling_factor
from mGPT.utils.temos_utils import subsample
from scripts.plys2npy import plys2npy
import random


# parsing argument
parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=32, help="input batch size")
parser.add_argument("--num_smplify_iters", type=int, default=100, help="num of smplify iterations")
parser.add_argument("--cuda", type=bool, default=True, help="enable cuda")
parser.add_argument("--gpu_ids", type=int, default=0, help="choose gpu ids")
parser.add_argument("--num_joints", type=int, default=22, help="joint number")
parser.add_argument("--joint_category", type=str, default="AMASS", help="use correspondence")
parser.add_argument("--fix_foot", type=str, default="False", help="fix foot or not")
parser.add_argument("--data_folder", type=str, default="", help="data folder")
parser.add_argument("--save_folder", type=str, default=None, help="results save folder")
parser.add_argument("--dir", type=str, default=None, help="folder to use")
parser.add_argument("--files", type=str, default="test_motion.npy", help="files to use")

opt = parser.parse_args()
print(opt)

# Set up device (CUDA if available)
device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")

# Load SMPL model
smplmodel = smplx.create(
    config.SMPL_MODEL_DIR,
    model_type="smpl",
    gender="neutral",
    ext="pkl",
    batch_size=opt.batchSize,
).to(device)

# Load the mean pose and shape
smpl_mean_file = config.SMPL_MEAN_FILE
file = h5py.File(smpl_mean_file, "r")
init_mean_pose = torch.from_numpy(file["pose"][:]).unsqueeze(0).float().repeat(opt.batchSize, 1).to(device)
init_mean_shape = torch.from_numpy(file["shape"][:]).unsqueeze(0).float().repeat(opt.batchSize, 1).to(device)
cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).unsqueeze(0).float().repeat(opt.batchSize, 1, 1).to(device)

# Initialize predicted parameters
pred_pose = torch.zeros(opt.batchSize, 72).to(device)
pred_betas = torch.zeros(opt.batchSize, 10).to(device)
pred_cam_t = torch.zeros(opt.batchSize, 1, 3).to(device)
keypoints_3d = torch.zeros(opt.batchSize, opt.num_joints, 3).to(device)

# Initialize SMPLify
smplify = SMPLify3D(
    smplxmodel=smplmodel,
    batch_size=opt.batchSize,
    joints_category=opt.joint_category,
    num_iters=opt.num_smplify_iters,
    device=device,
)
print("SMPLify3D initialized!")

# Collect files to process
paths = []
if opt.dir:
    file_list = sorted(os.listdir(opt.dir))
    # random.shuffle(file_list)
    for item in file_list:
        if item.endswith(".npy"):
            paths.append(os.path.join(opt.dir, item))
elif opt.files:
    paths.append(opt.files)

print(f"Begin processing {len(paths)} npy files!")

if not os.path.isdir(opt.save_folder):
    os.makedirs(opt.save_folder, exist_ok=True)

# Batch processing of npy files
for path in paths:

    # if path.split('/')[-1].split('_')[0] != '001355':
    #     continue

    dir_save = os.path.join(opt.save_folder, "results_smplfitting", "SMPLFit_" + os.path.basename(path)[:-4])
    if os.path.exists(join(opt.save_folder, "results_smplfitting", os.path.basename(path)[:-4] + "_mesh.npy")):
        print(f"npy is already processed: {path[:-4]}_mesh.npy")
        continue

    data = np.load(path)
    if len(data.shape) > 3:
        data = data[0]  # Handle multi-dimension data

    if data.shape[1] > 1000:
        print(f"Skipping mesh data: {dir_save}")
        continue

    print(f"Processing: {dir_save}")

    if not os.path.isdir(dir_save):
        os.makedirs(dir_save, exist_ok=True)

    if opt.num_joints == 22:
        frames = subsample(len(data), last_framerate=12.5, new_framerate=12.5)
        data = data[frames, ...]
    elif opt.num_joints == 21:
        frames = subsample(len(data), last_framerate=100, new_framerate=12.5)
        data = data.copy() * mmm_to_smplh_scaling_factor

    num_seqs = data.shape[0]

    pred_pose_prev = torch.zeros(opt.batchSize, 72).to(device)
    pred_betas_prev = torch.zeros(opt.batchSize, 10).to(device)
    pred_cam_t_prev = torch.zeros(opt.batchSize, 3).to(device)
    keypoints_3d_prev = torch.zeros(opt.batchSize, opt.num_joints, 3).to(device)

    for idx in range(0, num_seqs, opt.batchSize):

        batch_data = data[idx:idx + opt.batchSize]
        keypoints_3d[:len(batch_data)] = torch.Tensor(batch_data).to(device)

        # Set initial pose and shape
        if idx == 0:
            pred_betas[:len(batch_data)] = init_mean_shape[:len(batch_data)]
            pred_pose[:len(batch_data)] = init_mean_pose[:len(batch_data)]
            pred_cam_t[:len(batch_data)] = cam_trans_zero[:len(batch_data)]

        else:
            data_param = joblib.load(dir_save + "/" + "motion_%04d" % (idx - opt.batchSize) + ".pkl")
            pred_betas[:len(batch_data)] = torch.from_numpy(data_param["beta"]).float().to(device)[:len(batch_data)]
            pred_pose[:len(batch_data)] = torch.from_numpy(data_param["pose"]).float().to(device)[:len(batch_data)]
            pred_cam_t[:len(batch_data)] = torch.from_numpy(data_param["cam"]).float().to(device)[:len(batch_data)]

        if opt.joint_category == "AMASS":
            confidence_input = torch.ones(opt.num_joints)
            # make sure the foot and ankle
            if opt.fix_foot == True:
                confidence_input[7] = 1.5
                confidence_input[8] = 1.5
                confidence_input[10] = 1.5
                confidence_input[11] = 1.5
        elif opt.joint_category == "MMM":
            confidence_input = torch.ones(opt.num_joints)
        else:
            print("Such category not settle down!")

        # SMPL fitting
        (
            new_opt_vertices,
            new_opt_joints,
            new_opt_pose,
            new_opt_betas,
            new_opt_cam_t,
            new_opt_joint_loss,
        ) = smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input.to(device),
        )

        # Export results
        outputp = smplmodel(
            betas=new_opt_betas,
            global_orient=new_opt_pose[:, :3],
            body_pose=new_opt_pose[:, 3:],
            transl=new_opt_cam_t[:,0],
            return_verts=True,
        )

        param = {"beta": new_opt_betas.detach().cpu().numpy(), "pose": new_opt_pose.detach().cpu().numpy(), "cam": new_opt_cam_t.detach().cpu().numpy()}
        joblib.dump(param, os.path.join(dir_save, f"motion_{idx:04d}.pkl"), compress=3)

        for idx_save in range(len(batch_data)):
            mesh_p = trimesh.Trimesh(vertices=outputp.vertices[idx_save].detach().cpu().numpy().squeeze(), faces=smplmodel.faces, process=False)
            mesh_p.export(os.path.join(dir_save, f"motion_{idx+idx_save:04d}.ply"))

            # param = {"beta": new_opt_betas.detach().cpu().numpy(), "pose": new_opt_pose.detach().cpu().numpy(), "cam": new_opt_cam_t.detach().cpu().numpy()}
            # joblib.dump(param, os.path.join(dir_save, f"motion_{idx+idx_save:04d}.pkl"), compress=3)

    print(f"Finished processing {path}")
    plys2npy(dir_save, os.path.join(opt.save_folder, "results_smplfitting"))