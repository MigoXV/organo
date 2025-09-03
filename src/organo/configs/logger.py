from dataclasses import dataclass
from typing import Optional


@dataclass
class WandbConfig:
    name: Optional[str] = None
    save_dir: str = "."
    version: Optional[str] = None
    offline: bool = False
    dir: Optional[str] = None
    id: Optional[str] = None
    anonymous: Optional[bool] = None
    project: Optional[str] = None
    log_model: Optional[str] = None
    experiment: Optional[str] = None
    prefix: str = ""
    checkpoint_name: Optional[str] = None


@dataclass
class CSVConfig:
    save_dir: str = "."
    name: Optional[str] = "lightning_logs"
    version: Optional[str] = None
    prefix: str = ""
    flush_logs_every_n_steps: int = 100


@dataclass
class TensorBoardConfig:
    save_dir: str = "."
    name: Optional[str] = "lightning_logs"
    version: Optional[str] = None
    log_graph: bool = False
    default_hp_metric: bool = True
    prefix: str = ""
    sub_dir: Optional[str] = None
