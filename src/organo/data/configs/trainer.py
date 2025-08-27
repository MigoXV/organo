from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Union

# from lightning.pytorch.trainer.connectors.accelerator_connector import (
#     _LITERAL_WARN,
#     _PRECISION_INPUT,
# )


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    strategy: str = "auto"
    devices: str = "auto"
    num_nodes: int = 1
    # precision: Optional[_PRECISION_INPUT] = None
    # logger: Optional[Union[Logger, Iterable[Logger], bool]] = None
    # callbacks: Optional[Union[List[Callback], Callback]] = None
    fast_dev_run: int = 0
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = -1
    min_steps: Optional[int] = None
    max_time: Optional[str] = None
    # limit_train_batches: Optional[Union[int, float]] = None
    # limit_val_batches: Optional[Union[int, float]] = None
    # limit_test_batches: Optional[Union[int, float]] = None
    # limit_predict_batches: Optional[Union[int, float]] = None
    # overfit_batches: Union[int, float] = 0.0
    val_check_interval: Optional[int] = None
    check_val_every_n_epoch: Optional[int] = 1
    num_sanity_val_steps: Optional[int] = None
    log_every_n_steps: Optional[int] = None
    enable_checkpointing: Optional[bool] = None
    enable_progress_bar: Optional[bool] = None
    enable_model_summary: Optional[bool] = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None
    # deterministic: Optional[Union[bool, _LITERAL_WARN]] = None
    benchmark: Optional[bool] = None
    inference_mode: bool = True
    use_distributed_sampler: bool = True
    profiler: Optional[str] = None
    detect_anomaly: bool = False
    barebones: bool = False
    # plugins: Optional[Union[_PLUGIN_INPUT, List[_PLUGIN_INPUT]]] = None
    sync_batchnorm: bool = False
    reload_dataloaders_every_n_epochs: int = 0
    default_root_dir: Optional[str] = None
