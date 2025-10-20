import json
import os
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf
from lom.callback import build_callbacks
from lom.config import parse_args
from lom.data.build_data import build_data
from lom.models.build_model import build_model
from lom.utils.logger import create_logger
from lom.utils.load_checkpoint import load_pretrained, load_pretrained_vae, load_pretrained_lm, load_pretrained_vae_compositional

def print_table(title, metrics, logger=None):
    table = Table(title=title)

    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")

    logger.info(metrics) if logger else None


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER

    # Logger
    logger = create_logger(cfg, phase="test")
    logger.info(OmegaConf.to_yaml(cfg))

    # Output dir
    model_name = cfg.model.target.split('.')[-2].lower()
    output_dir = Path(
        os.path.join(cfg.FOLDER, model_name, cfg.NAME, "samples_" + cfg.TIME))
    if cfg.TEST.SAVE_PREDICTIONS:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving predictions to {str(output_dir)}")

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase="test")
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Model
    # model = build_model(cfg, datamodule)
    model = build_model(cfg)
    logger.info("model {} loaded".format(cfg.model.target))

    # Lightning Trainer
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        precision=cfg.TRAIN.PRECISION,
        accelerator=cfg.ACCELERATOR,
        devices=list(range(len(cfg.DEVICE))),
        default_root_dir=cfg.FOLDER_EXP,
        reload_dataloaders_every_n_epochs=1,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=None,
        callbacks=callbacks,
    )

    # Strict load vae model
    if OmegaConf.select(cfg.TRAIN, 'PRETRAINED_VQ') is not None or cfg.TEST.CHECKPOINTS_FACE:
        load_pretrained_vae_compositional(cfg, model, logger, phase="test")

    # Strict load pretrianed model
    if cfg.TEST.CHECKPOINTS:
        # load_pretrained_without_vqvae(cfg, model, logger, phase="test")
        load_pretrained_lm(cfg, model, logger, phase="test")


    metrics = trainer.test(model, datamodule=datamodule)

    print(metrics)



if __name__ == "__main__":
    main()
