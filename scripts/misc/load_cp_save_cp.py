import os
import glob
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from lom.callback import build_callbacks
from lom.config import parse_args, instantiate_from_config
from lom.data.build_data import build_data
from lom.models.build_model import build_model
from lom.utils.logger import create_logger
from lom.utils.load_checkpoint import load_pretrained, load_pretrained_vae, load_pretrained_without_vqvae
from utils_emage import other_tools

def main():
    # Configs
    cfg = parse_args(phase="train")  # parse config file

    # Logger
    logger = create_logger(cfg, phase="train")  # create logger
    logger.info(OmegaConf.to_yaml(cfg))  # print config file

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # # Environment Variables
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # # Set HTTP request timeout and maximum number of retries for WandB
    # os.environ['WANDB_INIT_TIMEOUT'] = '360'  # Extend the initialization timeout to 180 seconds

    # Metric Logger
    pl_loggers = []
    for loggerName in cfg.LOGGER.TYPE:
        if loggerName == 'tenosrboard' or cfg.LOGGER.WANDB.params.project:
            pl_logger = instantiate_from_config(
                eval(f'cfg.LOGGER.{loggerName.upper()}'))
            pl_loggers.append(pl_logger)

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase='train')
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Model
    model = build_model(cfg, datamodule)


    # # Strict load pretrianed model
    # if cfg.TRAIN.PRETRAINED:
    #     load_pretrained_without_vqvae(cfg, model, logger)

    # checkpoint_path = "/afs/cs.stanford.edu/u/juze/code/exp_motion/language_of_motion/experiments/lom/VQVAE_AMASS_upper_lower_papervision_debug/checkpoints/epoch=79.ckpt"
    # state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # print("State dict keys:", state_dict.keys())
    

    # checkpoint_path_mgpt = "/afs/cs.stanford.edu/u/juze/code/exp_motion/motiongpt/checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar"
    # state_dict_mgpt = torch.load(checkpoint_path_mgpt, map_location="cpu", weights_only=False)
    # Load and process all VAE checkpoints
    checkpoint_paths = {
        'face': "/afs/cs.stanford.edu/u/juze/code/exp_motion/language_of_motion/models/emage_vq/last_790_face_v2.bin",
        'hand': "/afs/cs.stanford.edu/u/juze/code/exp_motion/language_of_motion/models/emage_vq/hands_vertex_1layer_710.bin",
        'upper': "/afs/cs.stanford.edu/u/juze/code/exp_motion/language_of_motion/models/emage_vq/upper_vertex_1layer_710.bin",
        'lower': "/afs/cs.stanford.edu/u/juze/code/exp_motion/language_of_motion/models/emage_vq/lower_foot_600.bin",
        'global': "/afs/cs.stanford.edu/u/juze/code/exp_motion/language_of_motion/models/emage_vq/last_1700_foot.bin"
        
    }

    combined_state_dict = {}
    for part, path in checkpoint_paths.items():
        state_dict = torch.load(path, map_location="cpu", weights_only=False)['model_state']
        for key, value in state_dict.items():
            new_key = key.replace('module', f'vae_{part}')
            combined_state_dict[new_key] = value
    
    # Save the combined weights to a checkpoint file
    save_path = "/afs/cs.stanford.edu/u/juze/code/exp_motion/language_of_motion/models/pretrained_vq_emage/vq_emage_speaker_2.ckpt"
    torch.save({'state_dict': combined_state_dict}, save_path)
    print(f"Saved combined checkpoint to {save_path}")



# {'state_dict': new_state_dict}












#     model.load_state_dict(state_dict, strict=False)





if __name__ == "__main__":
    main()
