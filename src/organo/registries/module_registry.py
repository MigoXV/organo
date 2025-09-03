import typing

from class_registry import ClassRegistry
from hydra.core.config_store import ConfigStore

T = typing.TypeVar("T")


class ModuleRegistry:

    def __init__(self, registry_name: str):
        self.registry_name = registry_name
        self.module_registry = ClassRegistry(unique=True)
        self.dataclass_reistry = ClassRegistry(unique=True)

    def register(self, key: str, module_dataclass=None):

        def _decorator(module_cls):
            """Decorator to register a class."""
            self.module_registry.register(key)(module_cls)
            if module_dataclass is not None:
                self.dataclass_reistry.register(key)(module_dataclass)
                cs = ConfigStore.instance()
                cs.store(key, node=module_dataclass, group=self.registry_name)
            return module_cls

        return _decorator

    def __len__(self) -> int:
        return len(self.module_registry)

    def get_class(self, key: str):
        return self.module_registry.get_class(key), self.dataclass_reistry.get_class(
            key
        )

    def get_module_class(self, key: str):
        return self.module_registry.get_class(key)

    def get_config_class(self, key: str):
        return self.dataclass_reistry.get_class(key)

    def unregister(self, key: str) -> None:
        """Unregisters a class and its associated dataclass."""
        self.module_registry.unregister(key)
        self.dataclass_reistry.unregister(key)
