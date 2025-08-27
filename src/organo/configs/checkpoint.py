from dataclasses import dataclass
from typing import Optional

from lightning.fabric.utilities.types import _PATH


@dataclass
class CheckpointConfig:
    dirpath: Optional[_PATH] = None
    filename: Optional[str] = None
    monitor: Optional[str] = None
    verbose: bool = False
    save_last: Optional[str] = None
    save_top_k: int = 1
    save_weights_only: bool = False
    mode: str = "min"
    auto_insert_metric_name: bool = True
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[str] = None
    every_n_epochs: Optional[int] = None
    save_on_train_epoch_end: Optional[bool] = None
    enable_version_counter: bool = True
