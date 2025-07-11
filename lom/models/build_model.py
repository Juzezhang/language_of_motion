from omegaconf import OmegaConf
from lom.config import instantiate_from_config

# def build_model(cfg, datamodule):
def build_model(cfg):
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config['params']['cfg'] = cfg
    # model_config['params']['datamodule'] = datamodule
    return instantiate_from_config(model_config)
