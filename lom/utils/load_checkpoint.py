import torch
from torch.serialization import add_safe_globals
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata
from typing import Any

def load_pretrained(cfg, model, logger=None, phase="train"):
    if logger is not None:
        logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")
        
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
        
    # Add OmegaConf classes to safe globals before loading
    add_safe_globals([ListConfig, ContainerMetadata])
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    return model


def load_pretrained_debug(cfg, model, logger=None, phase="train"):
    from collections import OrderedDict
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
        
    # Add OmegaConf classes to safe globals before loading
    add_safe_globals([ListConfig, ContainerMetadata])
    states = torch.load(ckpt_path, weights_only=True, map_location=torch.device('cpu'))

    new_weights = OrderedDict()
    flag=False
    for k, v in states['model_state'].items():
        #print(k)
        if "module" not in k:
            break
        else:
            new_weights[k.replace('module','vae')]=v
            flag=True
    if flag:
        try:
            model.load_state_dict(new_weights, strict=False)
        except:
            #print(states['model_state'])
            model.load_state_dict(states['model_state'])
    else:
        model.load_state_dict(states['model_state'])

    return model


def load_pretrained_without_vqvae(cfg, model, logger=None, phase="train"):
    if logger is not None:
        logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")

    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS

    # Add all required classes to safe globals before loading
    # add_safe_globals([ListConfig, ContainerMetadata, Any])
    #if logger is not None:
    #    logger.info(f"Checkpoint keys: {list(model.keys())}")
    #else:
    #    print(f"Checkpoint keys: {list(model.keys())}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    # for bad in [
    # "lm.language_model.shared.weight",
    # "lm.language_model.encoder.embed_tokens.weight",
    # "lm.language_model.decoder.embed_tokens.weight",
    # "lm.language_model.lm_head.weight",
    # ]:
    #     state_dict.pop(bad, None)
    model.load_state_dict(state_dict, strict=False)
    return model

def load_pretrained_without_vae(cfg, model, logger=None, phase="train"):
    if logger is not None:
        logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")

    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS

    # Add OmegaConf classes to safe globals before loading
    add_safe_globals([ListConfig, ContainerMetadata])
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    return model

def load_pretrained_vae(cfg, model, logger, phase="train"):
    # Load pretrained VAE model
    if phase == "train" or phase == "token" or phase == "demo":
        checkpoint_path= cfg.TRAIN.PRETRAINED_VQ
    elif phase == "test":
        checkpoint_path = cfg.TEST.CHECKPOINTS
    # Load full checkpoint and extract only the state_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint['state_dict']  # Get only the state_dict
    
    # Create new state dict with modified keys
    state_dict_face = {}
    for key, value in state_dict.items():
        if 'vae_face' in key:
            new_key = key.replace('vae_face.', '')
            state_dict_face[new_key] = value
    
    state_dict_upper = {}
    for key, value in state_dict.items():
        if 'vae_upper' in key:
            new_key = key.replace('vae_upper.', '')
            state_dict_upper[new_key] = value
    
    state_dict_lower = {}
    for key, value in state_dict.items():
        if 'vae_lower' in key:
            new_key = key.replace('vae_lower.', '')
            state_dict_lower[new_key] = value
    
    state_dict_hand = {}
    for key, value in state_dict.items():
        if 'vae_hand' in key:
            new_key = key.replace('vae_hand.', '')
            state_dict_hand[new_key] = value
    
    state_dict_global = {}
    for key, value in state_dict.items():
        if 'vae_global' in key:
            new_key = key.replace('vae_global.', '')
            state_dict_global[new_key] = value
    
    # Save only the modified state_dict
    model.vae_face.load_state_dict(state_dict_face, strict=True)
    model.vae_upper.load_state_dict(state_dict_upper, strict=True)
    model.vae_lower.load_state_dict(state_dict_lower, strict=True)
    model.vae_hand.load_state_dict(state_dict_hand, strict=True)
    model.vae_global.load_state_dict(state_dict_global, strict=True)
    logger.info(f"Loaded pretrained VAE model from {checkpoint_path}")

    return model

def load_pretrained_vae_compositional(cfg, model, logger, phase="train"):
    # Load pretrained VAE model
    if phase == "train" or phase == "demo":
        # checkpoint_path= cfg.TRAIN.PRETRAINED_VQ
        checkpoint_path_face = cfg.TRAIN.PRETRAINED_FACE
        checkpoint_path_upper = cfg.TRAIN.PRETRAINED_UPPER
        checkpoint_path_lower = cfg.TRAIN.PRETRAINED_LOWER
        checkpoint_path_hand = cfg.TRAIN.PRETRAINED_HAND
        checkpoint_path_global = cfg.TRAIN.PRETRAINED_GLOBAL
    elif phase == "test" or phase == "token":
        checkpoint_path_face = cfg.TEST.CHECKPOINTS_FACE
        checkpoint_path_upper = cfg.TEST.CHECKPOINTS_UPPER
        checkpoint_path_lower = cfg.TEST.CHECKPOINTS_LOWER
        checkpoint_path_hand = cfg.TEST.CHECKPOINTS_HAND
        checkpoint_path_global = cfg.TEST.CHECKPOINTS_GLOBAL
        
    # Load full checkpoint and extract only the state_dict
    checkpoint_face = torch.load(checkpoint_path_face, map_location="cpu", weights_only=False)
    checkpoint_upper = torch.load(checkpoint_path_upper, map_location="cpu", weights_only=False)
    checkpoint_lower = torch.load(checkpoint_path_lower, map_location="cpu", weights_only=False)
    checkpoint_hand = torch.load(checkpoint_path_hand, map_location="cpu", weights_only=False)
    checkpoint_global = torch.load(checkpoint_path_global, map_location="cpu", weights_only=False)
    
    state_dict_face = checkpoint_face['state_dict']
    state_dict_upper = checkpoint_upper['state_dict']
    state_dict_lower = checkpoint_lower['state_dict']
    state_dict_hand = checkpoint_hand['state_dict']
    state_dict_global = checkpoint_global['state_dict']
    
    # Create new state dict with modified keys
    state_dict_face_new = {}
    for key, value in state_dict_face.items():
        if 'vae_face' in key:
            new_key = key.replace('vae_face.', '')
            state_dict_face_new[new_key] = value
    
    state_dict_upper_new = {}
    for key, value in state_dict_upper.items():
        if 'vae_upper' in key:
            new_key = key.replace('vae_upper.', '')
            state_dict_upper_new[new_key] = value
    
    state_dict_lower_new = {}
    for key, value in state_dict_lower.items():
        if 'vae_lower' in key:
            new_key = key.replace('vae_lower.', '')
            state_dict_lower_new[new_key] = value
    
    state_dict_hand_new = {}
    for key, value in state_dict_hand.items():
        if 'vae_hands' in key:
            new_key = key.replace('vae_hands.', '')
            state_dict_hand_new[new_key] = value
        elif 'vae_hand' in key:
            new_key = key.replace('vae_hand.', '')
            state_dict_hand_new[new_key] = value

    
    state_dict_global_new = {}
    for key, value in state_dict_global.items():
        if 'vae_global' in key:
            new_key = key.replace('vae_global.', '')
            state_dict_global_new[new_key] = value

    
    # Save only the modified state_dict
    model.vae_face.load_state_dict(state_dict_face_new, strict=True)
    model.vae_upper.load_state_dict(state_dict_upper_new, strict=True)
    model.vae_lower.load_state_dict(state_dict_lower_new, strict=True)
    model.vae_hand.load_state_dict(state_dict_hand_new, strict=True)
    model.vae_global.load_state_dict(state_dict_global_new, strict=True)
    # logger.info(f"Loaded pretrained VAE model from {checkpoint_path}")

    return model



def load_pretrained_vae_compositional_h3d(cfg, model, logger, phase="train"):
    # Load pretrained VAE model
    if phase == "train" or phase == "demo":
        # checkpoint_path= cfg.TRAIN.PRETRAINED_VQ
        checkpoint_path_face = cfg.TRAIN.PRETRAINED_FACE
        checkpoint_path_upper = cfg.TRAIN.PRETRAINED_UPPER
        checkpoint_path_lower = cfg.TRAIN.PRETRAINED_LOWER
        checkpoint_path_hand = cfg.TRAIN.PRETRAINED_HAND
        checkpoint_path_global = cfg.TRAIN.PRETRAINED_GLOBAL
    elif phase == "test" or phase == "token":
        checkpoint_path_face = cfg.TEST.CHECKPOINTS_FACE
        checkpoint_path_upper = cfg.TEST.CHECKPOINTS_UPPER
        checkpoint_path_lower = cfg.TEST.CHECKPOINTS_LOWER
        checkpoint_path_hand = cfg.TEST.CHECKPOINTS_HAND
        checkpoint_path_global = cfg.TEST.CHECKPOINTS_GLOBAL
        
    # Load full checkpoint and extract only the state_dict
    checkpoint_face = torch.load(checkpoint_path_face, map_location="cpu", weights_only=False)
    checkpoint_upper = torch.load(checkpoint_path_upper, map_location="cpu", weights_only=False)
    checkpoint_lower = torch.load(checkpoint_path_lower, map_location="cpu", weights_only=False)
    checkpoint_hand = torch.load(checkpoint_path_hand, map_location="cpu", weights_only=False)
    checkpoint_global = torch.load(checkpoint_path_global, map_location="cpu", weights_only=False)
    
    state_dict_face = checkpoint_face['state_dict']
    state_dict_upper = checkpoint_upper['state_dict']
    state_dict_lower = checkpoint_lower['state_dict']
    state_dict_hand = checkpoint_hand['state_dict']
    state_dict_global = checkpoint_global['state_dict']
    
    # Create new state dict with modified keys
    state_dict_face_new = {}
    for key, value in state_dict_face.items():
        if 'vae_face' in key:
            new_key = key.replace('vae_face.', '')
            state_dict_face_new[new_key] = value
    
    state_dict_upper_new = {}
    for key, value in state_dict_upper.items():
        if 'vae_upper' in key:
            new_key = key.replace('vae_upper.', '')
            state_dict_upper_new[new_key] = value
    
    # state_dict_lower_new = {}
    # for key, value in state_dict_lower.items():
    #     if 'vae_lower' in key:
    #         new_key = key.replace('vae_lower.', '')
    #         state_dict_lower_new[new_key] = value
    
    # Extract encoder/decoder
    from collections import OrderedDict
    vae_dict_lower = OrderedDict()
    for k, v in checkpoint_lower['state_dict'].items():
        if "motion_vae" in k:
            name = k.replace("motion_vae.", "")
            vae_dict_lower[name] = v
        elif "vae" in k:
            name = k.replace("vae.", "")
            vae_dict_lower[name] = v


    state_dict_hand_new = {}
    for key, value in state_dict_hand.items():
        if 'vae_hands' in key:
            new_key = key.replace('vae_hands.', '')
            state_dict_hand_new[new_key] = value
        elif 'vae_hand' in key:
            new_key = key.replace('vae_hand.', '')
            state_dict_hand_new[new_key] = value

    
    state_dict_global_new = {}
    for key, value in state_dict_global.items():
        if 'vae_global' in key:
            new_key = key.replace('vae_global.', '')
            state_dict_global_new[new_key] = value

    
    # Save only the modified state_dict
    model.vae_face.load_state_dict(state_dict_face_new, strict=True)
    model.vae_upper.load_state_dict(state_dict_upper_new, strict=True)
    # model.vae_lower.load_state_dict(state_dict_lower_new, strict=True)
    model.vae_lower.load_state_dict(vae_dict_lower, strict=True)
    model.vae_hand.load_state_dict(state_dict_hand_new, strict=True)
    model.vae_global.load_state_dict(state_dict_global_new, strict=True)
    # logger.info(f"Loaded pretrained VAE model from {checkpoint_path}")

    return model



def load_pretrained_lm(cfg, model, logger, phase="train"):
    # Load pretrained VAE model
    if phase == "train" or phase == "token":
        checkpoint_path= cfg.TRAIN.PRETRAINED_VQ
    elif phase == "test":
        checkpoint_path = cfg.TEST.CHECKPOINTS
    elif phase == "demo":
        checkpoint_path = cfg.TEST.CHECKPOINTS

    # Load full checkpoint and extract only the state_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint['state_dict']  # Get only the state_dict
    
    # Create new state dict with modified keys
    state_dict_lm = {}
    for key, value in state_dict.items():
        if 'lm.language_model' in key:
            new_key = key.replace('lm.language_model.', '')
            state_dict_lm[new_key] = value
        else:
            state_dict_lm[key] = value
    
    # Save only the modified state_dict
    model.lm.language_model.load_state_dict(state_dict_lm, strict=False)
    logger.info(f"Loaded pretrained LM model from {checkpoint_path}")

    return model


def load_pretrained_tokenizer(model, save_path):
    # Add OmegaConf classes to safe globals before loading
    add_safe_globals([ListConfig, ContainerMetadata])
    state_dict = torch.load(save_path,
                            map_location="cpu", weights_only=True)['state_dict']

    # Extract encoder/decoder
    from collections import OrderedDict
    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if "motion_vae" in k:
            name = k.replace("motion_vae.", "")
            vae_dict[name] = v
        elif "vae" in k:
            name = k.replace("vae.", "")
            vae_dict[name] = v
    if hasattr(model, 'vae'):
        model.load_state_dict(vae_dict, strict=True)
    else:
        model.load_state_dict(vae_dict, strict=True)

    return model
