"""Generic hook classes.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

import torch


class RemovableHandleList(list):
    """A list of handles that can be removed."""

    def clear(self):
        for handle in self:
            handle.remove()
        super().clear()


class HookType(str, Enum):
    """Enum for hook type."""

    FORWARD = "forward"
    BACKWARD = "backward"


class HookMode(str, Enum):
    """Enum for cache mode."""

    INPUT = "input"
    OUTPUT = "output"


@dataclass
class HookConfig(ABC):
    """
    Configuration for hooks.
    """

    hook_type: HookType = HookType.FORWARD
    hook_mode: HookMode = HookMode.OUTPUT
    module_exp: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    data_fn: Optional[Callable] = None


class Hook(ABC):
    """
    Abstract class for hooks.
    """

    def __init__(self, config: HookConfig):
        if not isinstance(config, HookConfig):
            raise ValueError(f"Expected HookConfig, got {type(config)}")
        self.config = config
        self.removable_handles = RemovableHandleList()
        self.storage: Dict[str, Any] = {}

    def register(self, module: torch.nn.Module):
        """
        Registers the hook.
        """
        if self.config.module_exp is None:
            compiled_exp = re.compile(r".*")
        else:
            compiled_exp = re.compile(self.config.module_exp)
        for name, module in module.named_modules():
            if name == "":
                continue
            if not compiled_exp.match(name):
                continue
            if self.config.hook_type is HookType.FORWARD:
                self.removable_handles.append(
                    module.register_forward_hook(self.forward_factory(name))
                )
            elif self.config.hook_type is HookType.BACKWARD:
                self.removable_handles.append(
                    module.register_backward_hook(self.backward_factory(name))
                )
            else:
                raise ValueError(f"Unknown hook type: {self.config.hook_type}")
        return self.removable_handles

    def remove(self):
        """Removes the hook."""
        self.removable_handles.clear()

    def clear(self):
        """Clears the storage and removes the hook."""
        self.storage.clear()
        self.removable_handles.clear()

    @abstractmethod
    def forward_factory(self, name: str):
        """
        Creates a hook factory.
        """
        pass

    @abstractmethod
    def backward_factory(self, name: str):
        """
        Creates a hook factory.
        """
        pass


class CacheHook(Hook):
    """
    Hook for caching.
    """

    def forward_factory(self, name: str):
        if self.config.hook_mode is HookMode.INPUT:

            def hook(module, input, output):
                self.storage[name] = input.detach()

        elif self.config.hook_mode is HookMode.OUTPUT:

            def hook(module, input, output):
                self.storage[name] = output.detach()

        else:
            raise ValueError(f"Unknown hook mode: {self.config.hook_mode}")
        return hook

    def backward_factory(self, name: str):
        raise NotImplementedError(
            "Backward hook not implemented for CacheHook"
        )


class MeasureHook(Hook):
    """
    Hook for measuring vectors.
    """

    def forward_factory(self, name: str):
        if self.config.data is not None:
            measure_data = self.config.data[name]
        else:
            measure_data = None
        if self.config.hook_mode is HookMode.INPUT:

            def hook(module, input, output):
                self.storage[name] = self.config.data_fn(
                    input.detach(), measure_data=measure_data
                )

        elif self.config.hook_mode is HookMode.OUTPUT:

            def hook(module, input, output):
                self.storage[name] = self.config.data_fn(
                    output.detach(), measure_data=measure_data
                )

        else:
            raise ValueError(f"Unknown hook mode: {self.config.hook_mode}")

        return hook

    def backward_factory(self, name: str):
        raise NotImplementedError(
            "Backward hook not implemented for MeasureHook"
        )


class ModifyHook(Hook):
    """
    Hook for modifying vectors.
    """

    def forward_factory(self, name: str):
        if self.config.data is not None:
            modify_data = self.config.data[name]
        else:
            modify_data = None
        if self.config.hook_mode is HookMode.INPUT:

            def hook(module, input, output):
                input = self.config.data_fn(input, modify_data=modify_data)
                return input

        elif self.config.hook_mode is HookMode.OUTPUT:

            def hook(module, input, output):
                output = self.config.data_fn(output, modify_data=modify_data)
                return output

        else:
            raise ValueError(f"Unknown hook mode: {self.config.hook_mode}")

        return hook

    def backward_factory(self, name: str):
        raise NotImplementedError(
            "Backward hook not implemented for ModifyHook"
        )
