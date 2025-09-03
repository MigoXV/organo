from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from organo.configs.logger import CSVConfig, TensorBoardConfig, WandbConfig
from organo.registers import logger_registry

logger_registry.register("wandb", WandbConfig)(WandbLogger)
logger_registry.register("csv", CSVConfig)(CSVLogger)
logger_registry.register("tensorboard", TensorBoardConfig)(TensorBoardLogger)
