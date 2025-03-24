import os
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm
from lom.config import parse_args
from lom.data.build_data import build_data
from lom.models.build_model import build_model
from loguru import logger
from lom.utils.load_checkpoint import load_pretrained_vae

def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = build_data(cfg, phase='token')
    print("datasets module initialized")

    # Load each dataset based on its type from the configuration
    for config in cfg.DATASET.datasets:
        dataset_name = config.get("name")
        code_path = config.get("code_path")
        if dataset_name == "AMASS":
            data_root_amass = cfg.DATASET["AMASS"].ROOT 
            output_dir_amass = os.path.join(data_root_amass, code_path)
            os.makedirs(output_dir_amass, exist_ok=True)
        if dataset_name == "BEAT2":
            data_root_beat2 = cfg.DATASET["BEAT2"].ROOT
            output_dir_beat2 = os.path.join(data_root_beat2, code_path)
            os.makedirs(output_dir_beat2, exist_ok=True)

    # Model
    model = build_model(cfg, datasets)
    logger.info("model {} loaded".format(cfg.model.target))

    load_pretrained_vae(cfg, model, logger, phase="token")

    if cfg.ACCELERATOR == "gpu":
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

    for batch in tqdm(datasets.token_dataloader(), desc=f'compositional motion tokenize'):

        seq_name =  batch["id_name"]
        dataset_name =  batch["dataset_name"][0]

        if dataset_name == 'amass':
            output_dir = output_dir_amass
        else:
            output_dir = output_dir_beat2

        tar_pose, tar_beta, tar_trans, tar_face, tar_hand, tar_upper, tar_lower = [
            batch[key].cuda() for key in ["pose", "shape", "trans", "face", "hand", "upper", "lower"]
        ]

        lower_dim = cfg.Representation_type.get('separate_rot').get('lower').get('vae_test_dim')
        bs, n = tar_pose.shape[0], tar_pose.shape[1]

        tar_index_value_face_top = model.vae_face.map2index(tar_face)  # bs*n/4
        tar_index_value_upper_top = model.vae_upper.map2index(tar_upper)  # bs*n/4
        tar_index_value_hands_top = model.vae_hand.map2index(tar_hand)  # bs*n/4
        tar_index_value_lower_top = model.vae_lower.map2index(tar_lower[..., :lower_dim])  # bs*n/4


        # rec_upper_test = model.vae_upper.decode(tar_index_value_upper_top.int())
        # rec_lower_test = model.vae_lower.decode(tar_index_value_lower_top.int())
        # rec_hands_test = model.vae_hand.decode(tar_index_value_hands_top.int())
        # rec_face_test = model.vae_face.decode(tar_index_value_face_top.int())

        # Save face code
        if dataset_name == 'beat2':
            target_path_face = os.path.join(output_dir,'face' , seq_name[0] + '.npy')
            Path(target_path_face).parent.mkdir(parents=True, exist_ok=True)
            np.save(target_path_face, tar_index_value_face_top.to('cpu').numpy())

        # Save upper code
        target_path_upper = os.path.join(output_dir, 'upper' , seq_name[0] + '.npy')
        Path(target_path_upper).parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path_upper, tar_index_value_upper_top.to('cpu').numpy())

        # Save hands code
        target_path_hands = os.path.join(output_dir, 'hands' , seq_name[0] + '.npy')
        Path(target_path_hands).parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path_hands, tar_index_value_hands_top.to('cpu').numpy())

        # Save lower code
        target_path_lower = os.path.join(output_dir, 'lower' , seq_name[0] + '.npy')
        Path(target_path_lower).parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path_lower, tar_index_value_lower_top.to('cpu').numpy())

    print(
        f'Motion tokenization done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()
