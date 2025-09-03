from dataclasses import dataclass
from typing import Any, Optional

from organo.configs.checkpoint import CheckpointConfig
from organo.configs.trainer import TrainerConfig


@dataclass
class MetaConfig:
    model: str
    task: str
    criterion: str
    optimizer: Optional[str] = None
    logger: Optional[str] = None


@dataclass
class TrainConfig:
    meta: MetaConfig
    checkpoint: CheckpointConfig = CheckpointConfig()
    trainer: TrainerConfig = TrainerConfig()
    model: Optional[Any] = None
    criterion: Optional[Any] = None
    task: Optional[Any] = None
    optimizer: Optional[Any] = None
    logger: Optional[Any] = None
