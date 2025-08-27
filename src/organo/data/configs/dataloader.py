from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Union

import torch
from torch.utils.data import Sampler


@dataclass
class DataLoaderConfig:
    batch_size: Optional[int] = 1
    shuffle: Optional[bool] = None
    sampler: Optional[Union["Sampler[Any]", Iterable[Any]]] = None
    batch_sampler: Optional[Union["Sampler[List[Any]]", Iterable[List[Any]]]] = None
    num_workers: int = 0
    collate_fn: Optional[Callable] = None
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0.0
    worker_init_fn: Optional[Callable] = None
    multiprocessing_context: Optional[str] = None
    generator: Optional["torch.Generator"] = None
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    pin_memory_device: str = ""
