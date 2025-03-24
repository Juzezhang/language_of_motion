import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from lom.utils.rotation_conversions import rotation_6d_to_matrix, rotation_6d_to_axis_angle, matrix_to_axis_angle
from lom.utils.other_tools import velocity2position, estimate_linear_velocity
from lom.data.mixed_dataset.data_tools import (
    joints_list, 
    JOINT_MASK_FACE,
    JOINT_MASK_UPPER,
    JOINT_MASK_HANDS,
    JOINT_MASK_LOWER,
    JOINT_MASK_FULL,
    BEAT_SMPLX_JOINTS,
    BEAT_SMPLX_FULL,
    BEAT_SMPLX_FACE,
    BEAT_SMPLX_UPPER,
    BEAT_SMPLX_HANDS,
    BEAT_SMPLX_LOWER
)

class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def compute_geodesic_distance(self, m1, m2):
        """ Compute the geodesic distance between two rotation matrices.

        Args:
            m1, m2: Two rotation matrices with the shape (batch x 3 x 3).

        Returns:
            The minimal angular difference between two rotation matrices in radian form [0, pi].
        """
        m1 = m1.reshape(-1, 3, 3)
        m2 = m2.reshape(-1, 3, 3)
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.clamp(cos, min=-1 + 1E-6, max=1-1E-6)

        theta = torch.acos(cos)

        return theta

    def __call__(self, m1, m2, reduction='mean'):
        loss = self.compute_geodesic_distance(m1, m2)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            raise RuntimeError(f'unsupported reduction: {reduction}')


class BCE_Loss(nn.Module):
    def __init__(self, args=None):
        super(BCE_Loss, self).__init__()
       
    def forward(self, fake_outputs, real_target):
        final_loss = F.cross_entropy(fake_outputs, real_target, reduce="mean")
        return final_loss

class weight_Loss(nn.Module):
    def __init__(self, args=None):
        super(weight_Loss, self).__init__()
    def forward(self, weight_f):
        weight_loss_div = torch.mean(weight_f[:, :, 0]*weight_f[:, :, 1])
        weight_loss_gap = torch.mean(-torch.log(torch.max(weight_f[:, :, 0], dim=1)[0] - torch.min(weight_f[:, :, 0], dim=1)[0]))
        return weight_loss_div, weight_loss_gap    
    

class HuberLoss(nn.Module):
    def __init__(self, beta=0.1, reduction="mean"):
        super(HuberLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, outputs, targets):
        final_loss = F.smooth_l1_loss(outputs / self.beta, targets / self.beta, reduction=self.reduction) * self.beta
        return final_loss
    

class KLDLoss(nn.Module):
    def __init__(self, beta=0.1):
        super(KLDLoss, self).__init__()
        self.beta = beta
    
    def forward(self, outputs, targets):
        final_loss = F.smooth_l1_loss((outputs / self.beta, targets / self.beta) * self.beta)
        return final_loss


class REGLoss(nn.Module):
    def __init__(self, beta=0.1):
        super(REGLoss, self).__init__()
        self.beta = beta
    
    def forward(self, outputs, targets):
        final_loss = F.smooth_l1_loss((outputs / self.beta, targets / self.beta) * self.beta)
        return final_loss    


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
    
    def forward(self, outputs, targets):
        final_loss = F.l2_loss(outputs, targets)
        return final_loss    






class UpperLoss(nn.Module):
    def __init__(self, Is_VQVAE=True):
        super(UpperLoss, self).__init__()
        self.rec_loss = GeodesicLoss()
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.vectices_loss = torch.nn.MSELoss(reduction='mean')
        self.Is_VQVAE = Is_VQVAE


    def inverse_selection_tensor(self, filtered_t, selection_array, n):
    # Create a zero array with shape n*165
        selection_array = torch.from_numpy(selection_array).to(filtered_t.device)
        original_shape_t = torch.zeros((n, 165)).to(filtered_t.device)
        
        # Find indices where selection array is 1
        selected_indices = torch.where(selection_array == 1)[0]
        
        # Fill the filtered_t values into original_shape_t at corresponding positions
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t


    def forward(self, rec_upper, tar_upper, tar_betas, tar_trans, Loss_6D, vertices_rec, vertices_tar):

        bs, n = tar_upper.shape[0], min(tar_upper.shape[1], rec_upper.shape[1])
        j = 13
        tar_exps = torch.zeros((bs, n, 100)).to(rec_upper.device)
        # tar_betas = torch.tile(tar_betas, (bs*n, 1))
        if Loss_6D == False:
            rec_upper = rotation_6d_to_matrix(rec_upper.reshape(bs, n, 13, 6))
            tar_upper = rotation_6d_to_matrix(tar_upper.reshape(bs, n, 13, 6))
        

        loss_rec_upper = self.rec_loss(rec_upper, tar_upper)

        velocity_loss = self.vel_loss(rec_upper[:, 1:] - rec_upper[:, :-1],
                                    tar_upper[:, 1:] - tar_upper[:, :-1])
        acceleration_loss = self.vel_loss(rec_upper[:, 2:] + rec_upper[:, :-2] - 2 * rec_upper[:, 1:-1],
                                        tar_upper[:, 2:] + tar_upper[:, :-2] - 2 * tar_upper[:, 1:-1])
            
        if Loss_6D == True:
            rec_upper = rotation_6d_to_axis_angle(rec_upper.reshape(bs, n, 13, 6)).reshape(bs*n, j*3)
            tar_upper = rotation_6d_to_axis_angle(tar_upper.reshape(bs, n, 13, 6)).reshape(bs*n, j*3)

        else:
            rec_upper = matrix_to_axis_angle(rec_upper.reshape(bs, n, 13, 3, 3)).reshape(bs*n, j*3)
            tar_upper = matrix_to_axis_angle(tar_upper.reshape(bs, n, 13, 3, 3)).reshape(bs*n, j*3)

        # rec_pose = self.inverse_selection_tensor(rec_upper, JOINT_MASK_UPPER, rec_upper.shape[0])
        # tar_pose = self.inverse_selection_tensor(tar_upper,  JOINT_MASK_UPPER, tar_upper.shape[0])
        # vertices_rec = smplx_func(
        #     betas=tar_betas.reshape(bs*n, 300), 
        #     transl=tar_trans.reshape(bs*n, 3), 
        #     expression=tar_exps.reshape(bs*n, 100), 
        #     jaw_pose=rec_pose[:, 66:69], 
        #     global_orient=rec_pose[:,:3], 
        #     body_pose=rec_pose[:,3:21*3+3], 
        #     left_hand_pose=rec_pose[:,25*3:40*3], 
        #     right_hand_pose=rec_pose[:,40*3:55*3], 
        #     return_verts=True,
        #     return_joints=True,
        #     leye_pose=tar_pose[:, 69:72], 
        #     reye_pose=tar_pose[:, 72:75],
        # )
        # vertices_tar = smplx_func(
        #     betas=tar_betas.reshape(bs*n, 300), 
        #     transl=tar_trans.reshape(bs*n, 3), 
        #     expression=tar_exps.reshape(bs*n, 100), 
        #     jaw_pose=tar_pose[:, 66:69], 
        #     global_orient=tar_pose[:,:3], 
        #     body_pose=tar_pose[:,3:21*3+3], 
        #     left_hand_pose=tar_pose[:,25*3:40*3], 
        #     right_hand_pose=tar_pose[:,40*3:55*3], 
        #     return_verts=True,
        #     return_joints=True,
        #     leye_pose=tar_pose[:, 69:72], 
        #     reye_pose=tar_pose[:, 72:75],
        # ) 

        vectices_loss = self.vectices_loss(vertices_rec['vertices'], vertices_tar['vertices'])

        vertices_vel_loss = self.vel_loss(vertices_rec['vertices'][:, 1:] - vertices_rec['vertices'][:, :-1], vertices_tar['vertices'][:, 1:] - vertices_tar['vertices'][:, :-1])
        vertices_acc_loss = self.vel_loss(vertices_rec['vertices'][:, 2:] + vertices_rec['vertices'][:, :-2] - 2 * vertices_rec['vertices'][:, 1:-1], vertices_tar['vertices'][:, 2:] + vertices_tar['vertices'][:, :-2] - 2 * vertices_tar['vertices'][:, 1:-1])

        final_loss = loss_rec_upper + velocity_loss + acceleration_loss + vectices_loss + vertices_vel_loss + vertices_acc_loss

        # if self.Is_VQVAE:
        #     final_loss += loss_embedding

        return final_loss    


class LowerLoss(nn.Module):
    def __init__(self, Is_VQVAE=True):
        super(LowerLoss, self).__init__()

        self.rec_loss = GeodesicLoss()
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.vectices_loss = torch.nn.MSELoss(reduction='mean')
        self.Is_VQVAE = Is_VQVAE

    def inverse_selection_tensor(self, filtered_t, selection_array, n):
    # Create a zero array with shape n*165
        selection_array = torch.from_numpy(selection_array).to(filtered_t.device)
        original_shape_t = torch.zeros((n, 165)).to(filtered_t.device)
        
        # Find indices where selection array is 1
        selected_indices = torch.where(selection_array == 1)[0]
        
        # Fill the filtered_t values into original_shape_t at corresponding positions
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t
    
    
    def forward(self, rec_lower, tar_lower, tar_betas, tar_trans, Loss_6D, vertices_rec, vertices_tar):

        bs, n = tar_lower.shape[0], min(tar_lower.shape[1], rec_lower.shape[1])
        j = 9

        tar_contact = tar_lower[:, :n, j*6+3:j*6+7]
        rec_contact = rec_lower[:, :n, j*6+3:j*6+7]
        tar_exps = torch.zeros((bs, n, 100)).to(rec_lower.device)
        # tar_betas = torch.tile(tar_betas, (n, 1))

        if Loss_6D == False:
            rec_lower = rotation_6d_to_matrix(rec_lower[..., :54].reshape(bs, n, 9, 6))
            tar_lower = rotation_6d_to_matrix(tar_lower[..., :54].reshape(bs, n, 9, 6))
        

        loss_rec_lower = self.rec_loss(rec_lower, tar_lower)
        loss_contact = self.vectices_loss(rec_contact, tar_contact)
        velocity_loss = self.vel_loss(rec_lower[:, 1:] - rec_lower[:, :-1],
                                    tar_lower[:, 1:] - tar_lower[:, :-1])
        acceleration_loss = self.vel_loss(rec_lower[:, 2:] + rec_lower[:, :-2] - 2 * rec_lower[:, 1:-1],
                                        tar_lower[:, 2:] + tar_lower[:, :-2] - 2 * tar_lower[:, 1:-1])
            

        if Loss_6D == True:
            rec_lower = rotation_6d_to_axis_angle(rec_lower.reshape(bs, n, 9, 6)).reshape(bs*n, j*3)
            tar_lower = rotation_6d_to_axis_angle(tar_lower.reshape(bs, n, 9, 6)).reshape(bs*n, j*3)

        else:
            rec_lower = matrix_to_axis_angle(rec_lower.reshape(bs, n, 9, 3, 3)).reshape(bs*n, j*3)
            tar_lower = matrix_to_axis_angle(tar_lower.reshape(bs, n, 9, 3, 3)).reshape(bs*n, j*3)


        # rec_pose = self.inverse_selection_tensor(rec_lower, JOINT_MASK_LOWER, rec_lower.shape[0])
        # tar_pose = self.inverse_selection_tensor(tar_lower, JOINT_MASK_LOWER, tar_lower.shape[0])
        # vertices_rec = smplx_func(
        #     betas=tar_betas.reshape(bs*n, 300), 
        #     transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
        #     expression=tar_exps.reshape(bs*n, 100), 
        #     jaw_pose=rec_pose[:, 66:69], 
        #     global_orient=rec_pose[:,:3], 
        #     body_pose=rec_pose[:,3:21*3+3], 
        #     left_hand_pose=rec_pose[:,25*3:40*3], 
        #     right_hand_pose=rec_pose[:,40*3:55*3], 
        #     return_verts=False,
        #     return_joints=True,
        #     leye_pose=tar_pose[:, 69:72], 
        #     reye_pose=tar_pose[:, 72:75],
        # )
        # vertices_tar = smplx_func(
        #     betas=tar_betas.reshape(bs*n, 300), 
        #     transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
        #     expression=tar_exps.reshape(bs*n, 100), 
        #     jaw_pose=tar_pose[:, 66:69], 
        #     global_orient=tar_pose[:,:3], 
        #     body_pose=tar_pose[:,3:21*3+3], 
        #     left_hand_pose=tar_pose[:,25*3:40*3], 
        #     right_hand_pose=tar_pose[:,40*3:55*3], 
        #     return_verts=False,
        #     return_joints=True,
        #     leye_pose=tar_pose[:, 69:72], 
        #     reye_pose=tar_pose[:, 72:75],
        # )  

        
        joints_rec = vertices_rec['joints']
        joints_rec = joints_rec.reshape(bs, n, -1, 3)
        vectices_loss = self.vectices_loss(vertices_rec['joints'], vertices_tar['joints'])


        
        foot_idx = [7, 8, 10, 11]
        # find static indices consistent with model's own predictions
        static_idx = rec_contact > 0.95  # N x S x 4
        model_feet = joints_rec[:, :, foot_idx]  # foot positions (N, S, 4, 3)
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # (N, S-1, 4, 3)
        model_foot_v[~static_idx] = 0
        foot_loss = self.vel_loss(
            model_foot_v, torch.zeros_like(model_foot_v)
        )

        final_loss = loss_rec_lower + velocity_loss + acceleration_loss + loss_contact + vectices_loss + foot_loss * 20


        # # ---------------------- vae -------------------------- #
        # if self.Is_VQVAE:
        #     final_loss += loss_embedding


        return final_loss




class GlobalLoss(nn.Module):
    def __init__(self, Is_VQVAE=False, pose_fps=30):
        super(GlobalLoss, self).__init__()
        self.rec_loss = GeodesicLoss()
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.vectices_loss = torch.nn.MSELoss(reduction='mean')
        self.Is_VQVAE = Is_VQVAE
        self.pose_fps = pose_fps

    def inverse_selection_tensor(self, filtered_t, selection_array, n):
    # Create a zero array with shape n*165
        selection_array = torch.from_numpy(selection_array).to(filtered_t.device)
        original_shape_t = torch.zeros((n, 165)).to(filtered_t.device)
        
        # Find indices where selection array is 1
        selected_indices = torch.where(selection_array == 1)[0]
        
        # Fill the filtered_t values into original_shape_t at corresponding positions
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t
    
    
    def forward(self, rec_global, tar_global, tar_betas, tar_trans, Loss_6D, vertices_rec, vertices_tar):

        bs, n = tar_global.shape[0], min(tar_global.shape[1], rec_global.shape[1])
        j = 9
        # tar_betas = torch.tile(tar_betas, (n, 1))
        tar_contact = tar_global[:, :n, j*6+3:j*6+7]
        rec_contact = rec_global[:, :n, j*6+3:j*6+7]
        tar_exps = torch.zeros((bs, n, 100)).to(rec_global.device)
        tar_trans_vel_x = estimate_linear_velocity(tar_trans[:, :, 0:1], dt=1/self.pose_fps)
        tar_trans_vel_z = estimate_linear_velocity(tar_trans[:, :, 2:3], dt=1/self.pose_fps)

        loss_contact = self.vectices_loss(rec_contact, tar_contact)

        rec_trans = rec_global[:, :, j*6:j*6+3]
        rec_x_trans = velocity2position(rec_trans[:, :, 0:1], 1/self.pose_fps, tar_trans[:, 0, 0:1])
        rec_z_trans = velocity2position(rec_trans[:, :, 2:3], 1/self.pose_fps, tar_trans[:, 0, 2:3])
        rec_y_trans = rec_trans[:,:,1:2]
        rec_xyz_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        loss_trans_vel = self.vel_loss(rec_trans[:, :, 0:1], tar_trans_vel_x) \
        + self.vel_loss(rec_trans[:, :, 2:3], tar_trans_vel_z)
        v3 =  self.vel_loss(rec_trans[:, :, 0:1][:, 1:] - rec_trans[:, :, 0:1][:, :-1], tar_trans_vel_x[:, 1:] - tar_trans_vel_x[:, :-1]) \
        + self.vel_loss(rec_trans[:, :, 2:3][:, 1:] - rec_trans[:, :, 2:3][:, :-1], tar_trans_vel_z[:, 1:] - tar_trans_vel_z[:, :-1]) 
        a3 = self.vel_loss(rec_trans[:, :, 0:1][:, 2:] + rec_trans[:, :, 0:1][:, :-2] - 2 * rec_trans[:, :, 0:1][:, 1:-1], tar_trans_vel_x[:, 2:] + tar_trans_vel_x[:, :-2] - 2 * tar_trans_vel_x[:, 1:-1]) \
        + self.vel_loss(rec_trans[:, :, 2:3][:, 2:] + rec_trans[:, :, 2:3][:, :-2] - 2 * rec_trans[:, :, 2:3][:, 1:-1], tar_trans_vel_z[:, 2:] + tar_trans_vel_z[:, :-2] - 2 * tar_trans_vel_z[:, 1:-1]) 
        # g_loss_final += 5*v3 
        # g_loss_final += 5*a3
        v2 =  self.vel_loss(rec_xyz_trans[:, 1:] - rec_xyz_trans[:, :-1], tar_trans[:, 1:] - tar_trans[:, :-1]) 
        a2 =  self.vel_loss(rec_xyz_trans[:, 2:] + rec_xyz_trans[:, :-2] - 2 * rec_xyz_trans[:, 1:-1], tar_trans[:, 2:] + tar_trans[:, :-2] - 2 * tar_trans[:, 1:-1]) 
        # g_loss_final += 5*v2 
        # g_loss_final += 5*a2 
        # g_loss_final += loss_trans_vel
        loss_trans = self.vel_loss(rec_xyz_trans, tar_trans)
        # g_loss_final += loss_trans


        rec_global = rotation_6d_to_axis_angle(rec_global[..., :54].reshape(bs, n, 9, 6)).reshape(bs*n, j*3)
        tar_global = rotation_6d_to_axis_angle(tar_global[..., :54].reshape(bs, n, 9, 6)).reshape(bs*n, j*3)


        # rec_pose = self.inverse_selection_tensor(rec_global, JOINT_MASK_LOWER, rec_global.shape[0])
        # tar_pose = self.inverse_selection_tensor(tar_global, JOINT_MASK_LOWER, tar_global.shape[0])
        # vertices_rec = smplx_func(  
        #     betas=tar_betas.reshape(bs*n, 300), 
        #     transl=rec_xyz_trans.reshape(bs*n, 3), 
        #     expression=tar_exps.reshape(bs*n, 100), 
        #     jaw_pose=rec_pose[:, 66:69], 
        #     global_orient=rec_pose[:,:3], 
        #     body_pose=rec_pose[:,3:21*3+3], 
        #     left_hand_pose=rec_pose[:,25*3:40*3], 
        #     right_hand_pose=rec_pose[:,40*3:55*3], 
        #     return_verts=True,
        #     return_joints=True,
        #     leye_pose=tar_pose[:, 69:72], 
        #     reye_pose=tar_pose[:, 72:75],
        # )
        # vertices_tar = smplx_func(
        #     betas=tar_betas.reshape(bs*n, 300), 
        #     transl=tar_trans.reshape(bs*n, 3), 
        #     expression=tar_exps.reshape(bs*n, 100), 
        #     jaw_pose=tar_pose[:, 66:69], 
        #     global_orient=tar_pose[:,:3], 
        #     body_pose=tar_pose[:,3:21*3+3], 
        #     left_hand_pose=tar_pose[:,25*3:40*3], 
        #     right_hand_pose=tar_pose[:,40*3:55*3], 
        #     return_verts=True,
        #     return_joints=True,
        #     leye_pose=tar_pose[:, 69:72], 
        #     reye_pose=tar_pose[:, 72:75],
        # )  
        joints_rec = vertices_rec['joints'] + rec_xyz_trans.reshape(bs*n, 3).unsqueeze(dim=1)
        vert_rec = vertices_rec['vertices'] + rec_xyz_trans.reshape(bs*n, 3).unsqueeze(dim=1)
        joints_tar = vertices_tar['joints'] + tar_trans.reshape(bs*n, 3).unsqueeze(dim=1)
        vert_tar = vertices_tar['vertices'] + tar_trans.reshape(bs*n, 3).unsqueeze(dim=1)

        # joints_rec = vertices_rec['joints']
        joints_rec = joints_rec.reshape(bs, n, -1, 3)
        vectices_loss = self.vectices_loss(vert_rec, vert_tar)
        vertices_vel_loss = self.vectices_loss(
            vert_rec[:, 1:] - vert_rec[:, :-1],
            vert_tar[:, 1:] - vert_tar[:, :-1])
        vertices_acc_loss = self.vectices_loss(
            vert_rec[:, 2:] + vert_rec[:, :-2] - 2 * vert_rec[:, 1:-1],
            vert_tar[:, 2:] + vert_tar[:, :-2] - 2 * vert_tar[:, 1:-1])
        foot_idx = [7, 8, 10, 11]
        # model_contact = rec_global[:, :, j*6+3:j*6+7]
        # find static indices consistent with model's own predictions
        static_idx = rec_contact > 0.95  # N x S x 4 
        model_feet = joints_rec[:, :, foot_idx]  # foot positions (N, S, 4, 3)
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # (N, S-1, 4, 3)
        model_foot_v[~static_idx] = 0
        foot_loss = self.vel_loss(
            model_foot_v, torch.zeros_like(model_foot_v)
        )

        final_loss = loss_contact +  5*v3 + 5*a3 + 5*v2 + 5*a2 + loss_trans_vel + loss_trans  + (vectices_loss+5*vertices_vel_loss+5*vertices_acc_loss) + foot_loss*20
        # loss_trans_related =  5*v3 + 5*a3 + 5*v2 + 5*a2

        return final_loss


class FaceLoss(nn.Module):
    def __init__(self, Is_VQVAE=True):
        super(FaceLoss, self).__init__()

        # self.rec_loss = GeodesicLoss()
        # self.vel_loss = torch.nn.L1Loss(reduction='mean')
        # self.vectices_loss = torch.nn.MSELoss(reduction='mean')

        self.rec_loss = GeodesicLoss()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.vel_loss = torch.nn.MSELoss(reduction='mean') #torch.nn.L1Loss(reduction='mean')
        self.vectices_loss = torch.nn.MSELoss(reduction='mean')
    

        self.Is_VQVAE = Is_VQVAE

    def inverse_selection_tensor(self, filtered_t, selection_array, n):
    # Create a zero array with shape n*165
        selection_array = torch.from_numpy(selection_array).to(filtered_t.device)
        original_shape_t = torch.zeros((n, 165)).to(filtered_t.device)
        
        # Find indices where selection array is 1
        selected_indices = torch.where(selection_array == 1)[0]
        
        # Fill the filtered_t values into original_shape_t at corresponding positions
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t
    
    
    def forward(self, rec_face, tar_face, tar_betas, tar_trans, Loss_6D, vertices_rec, vertices_tar):

        bs, n = tar_face.shape[0], min(tar_face.shape[1], rec_face.shape[1])
        j = 1


        if Loss_6D == False:
            rec_pose = rotation_6d_to_matrix(rec_face[:, :, :j*6].reshape(bs, n, j, 6))
            tar_pose = rotation_6d_to_matrix(tar_face[:, :, :j*6].reshape(bs, n, j, 6))

        loss_rec = self.rec_loss(rec_pose, tar_pose)
        # g_loss_final += loss_rec
        # jaw open 6d vel and acc loss
        velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1])  
        acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1]) 
        # g_loss_final += velocity_loss 
        # g_loss_final += acceleration_loss 
        # face parameter l1 loss
        rec_exps = rec_face[:, :, j*6:]
        tar_exps = tar_face[:, :, j*6:]
        loss_face = self.mse_loss(rec_exps, tar_exps) 
        # g_loss_final += loss_face
        # face parameter l1 vel and acc loss
        face_velocity_loss =  self.vel_loss(rec_exps[:, 1:] - rec_exps[:, :-1], tar_exps[:, 1:] - tar_exps[:, :-1])
        face_acceleration_loss =  self.vel_loss(rec_exps[:, 2:] + rec_exps[:, :-2] - 2 * rec_exps[:, 1:-1], tar_exps[:, 2:] + tar_exps[:, :-2] - 2 * tar_exps[:, 1:-1])
        # g_loss_final += face_velocity_loss
        # g_loss_final += face_acceleration_loss

            # vertices loss

        if Loss_6D == True:
            tar_pose = rotation_6d_to_axis_angle(tar_pose).reshape(bs*n, j*6)
            rec_pose = rotation_6d_to_axis_angle(rec_pose).reshape(bs*n, j*6)
        else:
            tar_pose = matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
            rec_pose = matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)

        # vertices_rec = smplx_func(
        #     betas=tar_betas.reshape(bs*n, 300), 
        #     transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
        #     expression=tar_exps.reshape(bs*n, 100), 
        #     jaw_pose=rec_pose, 
        #     global_orient=torch.zeros(bs*n, 3).to(rec_pose.device), 
        #     body_pose=torch.zeros(bs*n, 21*3).to(rec_pose.device), 
        #     left_hand_pose=torch.zeros(bs*n, 15*3).to(rec_pose.device), 
        #     right_hand_pose=torch.zeros(bs*n, 15*3).to(rec_pose.device), 
        #     return_verts=True,
        #     # return_joints=True,
        #     leye_pose=torch.zeros(bs*n, 3).to(rec_pose.device), 
        #     reye_pose=torch.zeros(bs*n, 3).to(rec_pose.device),
        # )
        # vertices_tar = smplx_func(
        #     betas=tar_betas.reshape(bs*n, 300), 
        #     transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
        #     expression=rec_exps.reshape(bs*n, 100), 
        #     jaw_pose=tar_pose, 
        #     global_orient=torch.zeros(bs*n, 3).to(rec_pose.device), 
        #     body_pose=torch.zeros(bs*n, 21*3).to(rec_pose.device), 
        #     left_hand_pose=torch.zeros(bs*n, 15*3).to(rec_pose.device), 
        #     right_hand_pose=torch.zeros(bs*n, 15*3).to(rec_pose.device), 
        #     return_verts=True,
        #     # return_joints=True,
        #     leye_pose=torch.zeros(bs*n, 3).to(rec_pose.device), 
        #     reye_pose=torch.zeros(bs*n, 3).to(rec_pose.device),
        # )


        vectices_loss = self.mse_loss(vertices_rec['vertices'], vertices_tar['vertices'])
        # g_loss_final += vectices_loss
        # vertices vel and acc loss
        vert_velocity_loss =  self.vel_loss(vertices_rec['vertices'][:, 1:] - vertices_rec['vertices'][:, :-1], vertices_tar['vertices'][:, 1:] - vertices_tar['vertices'][:, :-1]) 
        vert_acceleration_loss =  self.vel_loss(vertices_rec['vertices'][:, 2:] + vertices_rec['vertices'][:, :-2] - 2 * vertices_rec['vertices'][:, 1:-1], vertices_tar['vertices'][:, 2:] + vertices_tar['vertices'][:, :-2] - 2 * vertices_tar['vertices'][:, 1:-1]) 
        # g_loss_final += vert_velocity_loss
        # g_loss_final += vert_acceleration_loss
        
        # # ---------------------- vae -------------------------- #
        # if "VQVAE" in self.args.g_name:
        #     loss_embedding = net_out["embedding_loss"]
        #     g_loss_final += loss_embedding
        #     self.tracker.update_meter("com", "train", loss_embedding.item())

        final_loss = loss_rec + velocity_loss + acceleration_loss + loss_face + face_velocity_loss + face_acceleration_loss + vectices_loss + vert_velocity_loss + vert_acceleration_loss


        # # ---------------------- vae -------------------------- #
        # if self.Is_VQVAE:
        #     final_loss += loss_embedding


        return final_loss


class HandLoss(nn.Module):
    def __init__(self, Is_VQVAE=True):
        super(HandLoss, self).__init__()
        self.rec_loss = GeodesicLoss()
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.vectices_loss = torch.nn.MSELoss(reduction='mean')
        self.Is_VQVAE = Is_VQVAE


    def inverse_selection_tensor(self, filtered_t, selection_array, n):
    # Create a zero array with shape n*165
        selection_array = torch.from_numpy(selection_array).to(filtered_t.device)
        original_shape_t = torch.zeros((n, 165)).to(filtered_t.device)
        
        # Find indices where selection array is 1
        selected_indices = torch.where(selection_array == 1)[0]
        
        # Fill the filtered_t values into original_shape_t at corresponding positions
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t


    def forward(self, rec_hand, tar_hand, tar_betas, tar_trans, Loss_6D, vertices_rec, vertices_tar):

        bs, n = tar_hand.shape[0], min(tar_hand.shape[1], rec_hand.shape[1])
        j = 30
        tar_exps = torch.zeros((bs, n, 100)).to(rec_hand.device)
        # tar_betas = torch.tile(tar_betas, (bs*n, 1))
        if Loss_6D == False:
            rec_hand = rotation_6d_to_matrix(rec_hand.reshape(bs, n, j, 6))
            tar_hand = rotation_6d_to_matrix(tar_hand.reshape(bs, n, j, 6))
        

        loss_rec_hand = self.rec_loss(rec_hand, tar_hand)

        velocity_loss = self.vel_loss(rec_hand[:, 1:] - rec_hand[:, :-1],
                                    tar_hand[:, 1:] - tar_hand[:, :-1])
        acceleration_loss = self.vel_loss(rec_hand[:, 2:] + rec_hand[:, :-2] - 2 * rec_hand[:, 1:-1],
                                        tar_hand[:, 2:] + tar_hand[:, :-2] - 2 * tar_hand[:, 1:-1])
            
        if Loss_6D == True:
            rec_hand = rotation_6d_to_axis_angle(rec_hand.reshape(bs, n, j, 6)).reshape(bs*n, j*3)
            tar_hand = rotation_6d_to_axis_angle(tar_hand.reshape(bs, n, j, 6)).reshape(bs*n, j*3)

        else:
            rec_hand = matrix_to_axis_angle(rec_hand.reshape(bs, n, j, 3, 3)).reshape(bs*n, j*3)
            tar_hand = matrix_to_axis_angle(tar_hand.reshape(bs, n, j, 3, 3)).reshape(bs*n, j*3)

        # rec_pose = self.inverse_selection_tensor(rec_hand, JOINT_MASK_HANDS, rec_hand.shape[0])
        # tar_pose = self.inverse_selection_tensor(tar_hand, JOINT_MASK_HANDS, tar_hand.shape[0])
        # vertices_rec = smplx_func(
        #     betas=tar_betas.reshape(bs*n, 300), 
        #     transl=tar_trans.reshape(bs*n, 3), 
        #     expression=tar_exps.reshape(bs*n, 100), 
        #     jaw_pose=rec_pose[:, 66:69], 
        #     global_orient=rec_pose[:,:3], 
        #     body_pose=rec_pose[:,3:21*3+3], 
        #     left_hand_pose=rec_pose[:,25*3:40*3], 
        #     right_hand_pose=rec_pose[:,40*3:55*3], 
        #     return_verts=True,
        #     return_joints=True,
        #     leye_pose=tar_pose[:, 69:72], 
        #     reye_pose=tar_pose[:, 72:75],
        # )
        # vertices_tar = smplx_func(
        #     betas=tar_betas.reshape(bs*n, 300), 
        #     transl=tar_trans.reshape(bs*n, 3), 
        #     expression=tar_exps.reshape(bs*n, 100), 
        #     jaw_pose=tar_pose[:, 66:69], 
        #     global_orient=tar_pose[:,:3], 
        #     body_pose=tar_pose[:,3:21*3+3], 
        #     left_hand_pose=tar_pose[:,25*3:40*3], 
        #     right_hand_pose=tar_pose[:,40*3:55*3], 
        #     return_verts=True,
        #     return_joints=True,
        #     leye_pose=tar_pose[:, 69:72], 
        #     reye_pose=tar_pose[:, 72:75],
        # )  
        vectices_loss = self.vectices_loss(vertices_rec['vertices'], vertices_tar['vertices'])

        vertices_vel_loss = self.vel_loss(vertices_rec['vertices'][:, 1:] - vertices_rec['vertices'][:, :-1], vertices_tar['vertices'][:, 1:] - vertices_tar['vertices'][:, :-1])
        vertices_acc_loss = self.vel_loss(vertices_rec['vertices'][:, 2:] + vertices_rec['vertices'][:, :-2] - 2 * vertices_rec['vertices'][:, 1:-1], vertices_tar['vertices'][:, 2:] + vertices_tar['vertices'][:, :-2] - 2 * vertices_tar['vertices'][:, 1:-1])

        final_loss = loss_rec_hand + velocity_loss + acceleration_loss + vectices_loss + vertices_vel_loss + vertices_acc_loss

        # if self.Is_VQVAE:
        #     final_loss += loss_embedding

        return final_loss    


LOSS_FUNC_LUT = {
        "bce_loss": BCE_Loss,
        "l2_loss": L2Loss,
        "huber_loss": HuberLoss,
        "kl_loss": KLDLoss,
        "id_loss": REGLoss,
        "GeodesicLoss": GeodesicLoss,
        "weight_Loss": weight_Loss,
        "UpperLoss": UpperLoss,
        "LowerLoss": LowerLoss,
        "GlobalLoss": GlobalLoss,
        "FaceLoss": FaceLoss,
        "HandLoss": HandLoss,
    }


def get_loss_func(loss_name, **kwargs):    
    loss_func_class = LOSS_FUNC_LUT.get(loss_name)   
    loss_func = loss_func_class(**kwargs)   
    return loss_func


