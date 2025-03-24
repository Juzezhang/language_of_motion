import torch
import smplx
import os
import numpy as np
from omegaconf import OmegaConf
import logging

# Global SMPLX model instances - will be initialized on first use
_SMPLX_MODELS = {}

def get_smplx_model(model_type='smplx', model_path=None, cfg=None, device='cuda'):
    """
    Get or create a singleton SMPLX model instance.
    
    Args:
        model_type (str): Type of model ('smplx' or 'smplx2020')
        model_path (str, optional): Path to SMPLX model folder.
        cfg (OmegaConf, optional): Configuration object containing SMPLX settings
        device (str, optional): Device to place the model on.
        
    Returns:
        smplx.SMPLX: The SMPLX model instance
    """
    global _SMPLX_MODELS
    
    # Create a unique key for this model configuration
    model_key = f"{model_type}_{device}"
    
    if model_key not in _SMPLX_MODELS:
        # Determine the model path from various sources (prioritize config)
        if model_path is None:
            if cfg is not None and hasattr(cfg, 'DATASET') and hasattr(cfg.DATASET, 'SMPL_MODEL_DIR'):
                model_path = cfg.DATASET.SMPL_MODEL_DIR
            else:
                model_path = os.environ.get('SMPLX_MODEL_DIR', './models/smpl_models/smplx')
        
        logging.info(f"Initializing {model_type} model from {model_path}")
        
        # Configure model based on type
        if model_type == 'smplx2020':
            _SMPLX_MODELS[model_key] = smplx.create(
                model_path=model_path,
                model_type='smplx',
                gender='neutral',
                use_face_contour=True,
                num_betas=300,
                num_expression_coeffs=100,
                ext='npz',
                use_pca=False
            ).to(device)
        elif model_type == 'smplx':
            _SMPLX_MODELS[model_key] = smplx.create(
                model_path=model_path,
                model_type='smplx',
                gender='neutral',
                use_face_contour=True,
                num_betas=10,
                num_expression_coeffs=10,
                ext='npz',
                use_pca=True
            ).to(device)
        else:
            raise ValueError(f"Unknown SMPLX model type: {model_type}")
    
    return _SMPLX_MODELS[model_key]

def apply_smplx2020(
    betas=None,
    transl=None,
    expression=None,
    jaw_pose=None, 
    global_orient=None,
    body_pose=None,
    left_hand_pose=None,
    right_hand_pose=None,
    leye_pose=None,
    reye_pose=None,
    return_joints=True,
    cfg=None,
    device='cuda'
):
    """
    Apply SMPLX 2020 model (300 betas, 100 expressions, no PCA)
    
    Args:
        betas (tensor): Shape parameters (300 dimensions)
        transl (tensor): Translation parameters (3 dimensions)
        expression (tensor): Expression parameters (100 dimensions)
        jaw_pose (tensor): Jaw pose parameters (3 dimensions)
        global_orient (tensor): Global orientation parameters (3 dimensions)
        body_pose (tensor): Body pose parameters (21*3 dimensions)
        left_hand_pose (tensor): Left hand pose parameters (15*3 dimensions)
        right_hand_pose (tensor): Right hand pose parameters (15*3 dimensions)
        leye_pose (tensor): Left eye pose parameters (3 dimensions)
        reye_pose (tensor): Right eye pose parameters (3 dimensions)
        return_joints (bool): Whether to return joints
        cfg (OmegaConf, optional): Configuration object with path to model
        device (str): Device to use
        
    Returns:
        dict: SMPLX output
    """
    model = get_smplx_model(model_type='smplx2020', cfg=cfg, device=device)
    
    # Handle default parameters if None
    if betas is None:
        betas = torch.zeros(1, 300, device=device)
    if transl is None:
        transl = torch.zeros(1, 3, device=device)
    if expression is None:
        expression = torch.zeros(1, 100, device=device)
    
    # Apply the model
    output = model(
        betas=betas,
        transl=transl,
        expression=expression,
        jaw_pose=jaw_pose,
        global_orient=global_orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        return_joints=return_joints
    )
    
    return output

def apply_smplx(
    betas=None,
    transl=None,
    expression=None,
    jaw_pose=None, 
    global_orient=None,
    body_pose=None,
    left_hand_pose=None,
    right_hand_pose=None,
    leye_pose=None,
    reye_pose=None,
    return_joints=True,
    cfg=None,
    device='cuda'
):
    """
    Apply standard SMPLX model (10 betas, 10 expressions, PCA)
    
    Args:
        betas (tensor): Shape parameters (10 dimensions)
        transl (tensor): Translation parameters (3 dimensions)
        expression (tensor): Expression parameters (10 dimensions)
        jaw_pose (tensor): Jaw pose parameters (3 dimensions)
        global_orient (tensor): Global orientation parameters (3 dimensions)
        body_pose (tensor): Body pose parameters (21*3 dimensions)
        left_hand_pose (tensor): Left hand pose parameters (15*3 dimensions or 45 if not using PCA)
        right_hand_pose (tensor): Right hand pose parameters (15*3 dimensions or 45 if not using PCA)
        leye_pose (tensor): Left eye pose parameters (3 dimensions)
        reye_pose (tensor): Right eye pose parameters (3 dimensions)
        return_joints (bool): Whether to return joints
        cfg (OmegaConf, optional): Configuration object with path to model
        device (str): Device to use
        
    Returns:
        dict: SMPLX output
    """
    model = get_smplx_model(model_type='smplx', cfg=cfg, device=device)
    
    # Handle default parameters if None
    if betas is None:
        betas = torch.zeros(1, 10, device=device)
    if transl is None:
        transl = torch.zeros(1, 3, device=device)
    if expression is None:
        expression = torch.zeros(1, 10, device=device)
    
    # Apply the model
    output = model(
        betas=betas,
        transl=transl,
        expression=expression,
        jaw_pose=jaw_pose,
        global_orient=global_orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        return_joints=return_joints
    )
    
    return output

# Convenience functions for specific model types
def apply_smplx_standard(*args, **kwargs):
    """Apply standard SMPLX model"""
    kwargs['model_type'] = 'smplx'
    return apply_smplx(*args, **kwargs)

def apply_smplx2020(*args, **kwargs):
    """Apply SMPLX 2020 model"""
    kwargs['model_type'] = 'smplx2020'
    return apply_smplx2020(*args, **kwargs) 