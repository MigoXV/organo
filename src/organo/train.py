from typing import Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from organo.configs.checkpoint import CheckpointConfig
from organo.registers import logger_registry, task_registry

torch.set_float32_matmul_precision("high")


def get_config(config_dataclass, config: Optional[OmegaConf] = None) -> DictConfig:
    default_config = OmegaConf.structured(config_dataclass)
    if config is not None:
        tgt_config = OmegaConf.merge(default_config, config)
        assert isinstance(
            tgt_config, DictConfig
        ), "Merged configuration is not of type DictConfig"
    return tgt_config


def train(config: DictConfig):
    # 构建Lightning Datamodule
    task_cls = task_registry.get_module_class(config.meta.task)
    task = task_cls(config.task)
    task.build_model(config.meta.model, config.model)
    task.build_criterion(config.meta.criterion, config.criterion)
    # 构建Checkpoint回调
    ckpt_callback = ModelCheckpoint(**get_config(CheckpointConfig, config.checkpoint))
    # 构建logger
    logger = None
    if config.meta.logger is not None:
        logger_cls = logger_registry.get_module_class(config.meta.logger)
        logger = logger_cls(**config.logger)
    trainer = L.Trainer(
        default_root_dir="outputs/test01",
        devices=[0],
        # accelerator="cpu",
        callbacks=[ckpt_callback],
        logger=logger,
        log_every_n_steps=5,
    )
    data_module = task.build_data_module(
        batch_size=8,
        num_workers=0,
    )
    trainer.fit(task, datamodule=data_module)
