from typing import Optional

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from organo.configs.config import TrainConfig


def init_config_store(
    model: Optional[str] = None,
    task: Optional[str] = None,
    criterion: Optional[str] = None,
    optimizer: Optional[str] = None,
    logger: Optional[str] = None,
) -> DictConfig:
    cs = ConfigStore.instance()
    cs.store(name="train", node=TrainConfig)
    overrides = []
    if model is not None:
        overrides.append(f"+model={model}")
    if task is not None:
        overrides.append(f"+task={task}")
    if criterion is not None:
        overrides.append(f"+criterion={criterion}")
    if optimizer is not None:
        overrides.append(f"+optimizer={optimizer}")
    if logger is not None:
        overrides.append(f"+logger={logger}")
    with initialize(version_base=None, job_name="inmem"):
        cfg = compose(config_name="train", overrides=overrides)
    return cfg
