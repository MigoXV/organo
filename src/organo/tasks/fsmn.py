from typing import Union

import lightning as L
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn import Module


class FSMNVAD(L.LightningModule):

    @classmethod
    def build_task(cls, cfg: OmegaConf):
        task = cls(cfg)
        return task

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.datasets = {}

        self.model: Union[Module, None] = None
        self.criterion: Union[Module, None] = None

    def training_step(self, batch: Tensor, batch_idx: int):
        raise NotImplementedError(
            f"{self.__class__.__name__}.training_step not implemented"
        )

    def validation_step(self, batch: Tensor, batch_idx: int):
        raise NotImplementedError(
            f"{self.__class__.__name__}.validation_step not implemented"
        )

    def build_data_module(self, batch_size: int, num_workers: int = 0):
        raise NotImplementedError(
            f"{self.__class__.__name__}.build_data_module not implemented"
        )

    def build_model(self, key: str, cfg: OmegaConf):
        raise NotImplementedError(
            f"{self.__class__.__name__}.build_model not implemented"
        )

    def build_criterion(self, key: str, cfg: OmegaConf):
        raise NotImplementedError(
            f"{self.__class__.__name__}.build_criterion not implemented"
        )
