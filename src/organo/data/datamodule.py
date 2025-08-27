import inspect
from collections.abc import Iterable
from typing import Any, Iterable, Optional, Union

import lightning
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from lightning_utilities import apply_to_collection
from torch.utils.data import DataLoader, Dataset

from organo.configs.dataloader import DataLoaderConfig


class OrganoDataModule(lightning.LightningDataModule):
    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        val_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        predict_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        train_dataloader_config: Optional[DataLoaderConfig] = None,
        val_dataloader_config: Optional[DataLoaderConfig] = None,
        test_dataloader_config: Optional[DataLoaderConfig] = None,
        predict_dataloader_config: Optional[DataLoaderConfig] = None,
        **datamodule_kwargs: Any,
    ) -> "OrganoDataModule":

        def dataloader(
            ds: Dataset, dataloader_config: Optional[DataLoaderConfig] = None
        ) -> DataLoader:
            return DataLoader(ds, **dataloader_config.__dict__)

        def train_dataloader() -> TRAIN_DATALOADERS:
            return apply_to_collection(
                train_dataset,
                Dataset,
                dataloader,
                dataloader_config=train_dataloader_config,
            )

        def val_dataloader() -> EVAL_DATALOADERS:
            return apply_to_collection(
                val_dataset,
                Dataset,
                dataloader,
                dataloader_config=val_dataloader_config,
            )

        def test_dataloader() -> EVAL_DATALOADERS:
            return apply_to_collection(
                test_dataset,
                Dataset,
                dataloader,
                dataloader_config=test_dataloader_config,
            )

        def predict_dataloader() -> EVAL_DATALOADERS:
            return apply_to_collection(
                predict_dataset,
                Dataset,
                dataloader,
                dataloader_config=predict_dataloader_config,
            )

        candidate_kwargs = {
            "train_dataloader_config": train_dataloader_config,
            "val_dataloader_config": val_dataloader_config,
            "test_dataloader_config": test_dataloader_config,
            "predict_dataloader_config": predict_dataloader_config,
        }
        accepted_params = inspect.signature(cls.__init__).parameters
        accepts_kwargs = any(
            param.kind == param.VAR_KEYWORD for param in accepted_params.values()
        )
        if accepts_kwargs:
            special_kwargs = candidate_kwargs
        else:
            accepted_param_names = set(accepted_params)
            accepted_param_names.discard("self")
            special_kwargs = {
                k: v for k, v in candidate_kwargs.items() if k in accepted_param_names
            }

        datamodule = cls(**datamodule_kwargs, **special_kwargs)
        if train_dataset is not None:
            datamodule.train_dataloader = train_dataloader  # type: ignore[method-assign]
        if val_dataset is not None:
            datamodule.val_dataloader = val_dataloader  # type: ignore[method-assign]
        if test_dataset is not None:
            datamodule.test_dataloader = test_dataloader  # type: ignore[method-assign]
        if predict_dataset is not None:
            datamodule.predict_dataloader = predict_dataloader  # type: ignore[method-assign]
        return datamodule
