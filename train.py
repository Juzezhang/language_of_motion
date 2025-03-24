# import os
# import glob
# import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from lom.callback import build_callbacks
from lom.config import parse_args, instantiate_from_config
from lom.data.build_data import build_data
from lom.models.build_model import build_model
from lom.utils.logger import create_logger
from lom.utils.load_checkpoint import load_pretrained, load_pretrained_vae, load_pretrained_without_vqvae
# from utils_emage import other_tools

def main():
    # Configs
    cfg = parse_args(phase="train")  # parse config file

    # Logger
    logger = create_logger(cfg, phase="train")  # create logger
    logger.info(OmegaConf.to_yaml(cfg))  # print config file

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

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
    # if cfg.TRAIN.FORCE_BF16 and cfg.TRAIN.PRECISION == 'bf16':
    #     model.to(torch.bfloat16)  # convert model weight to BF16

    logger.info("model {} loaded".format(cfg.model.target))

    # Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        precision=cfg.TRAIN.PRECISION,
        logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp_find_unused_parameters_true"
        if len(cfg.DEVICE) > 1 else 'auto',
        benchmark=False,
        deterministic=False,
        # num_sanity_val_steps=0
    )
    logger.info("Trainer initialized")

    # Strict load pretrianed model
    if cfg.TRAIN.PRETRAINED:
        load_pretrained_without_vqvae(cfg, model, logger)

    # Strict load vae model
    if OmegaConf.select(cfg.TRAIN, 'PRETRAINED_VQ') is not None:
        load_pretrained_vae(cfg, model, logger)


    # Pytorch 2.0 Compile
    # if torch.__version__ >= "2.0.0":
    #     model = torch.compile(model, mode="reduce-overhead")
    # model = torch.compile(model)

    # Lightning Fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule=datamodule,
                    ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datamodule)

    # Training ends
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")



if __name__ == "__main__":
    main()
