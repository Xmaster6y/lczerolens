"""
Generic lens class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

import chess
import torch

from lczerolens.adapt import ModelWrapper


class Lens(ABC):
    """
    Generic lens class.
    """

    @abstractmethod
    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        pass

    @abstractmethod
    def compute_heatmap(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Computes the heatmap for a given board.
        """
        pass


class RemovableHandleList(list):
    """
    A list of handles that can be removed.
    """

    def remove(self):
        for handle in self:
            handle.remove()
        self.clear()


class HookArgs(type):
    """
    Metaclass for hook arguments.
    """

    pass


class HookFactory(ABC):
    """
    Abstract class for hook factories.
    """

    def __init__(self):
        """
        Initializes the hook factory.
        """
        self.removable_handles = RemovableHandleList()

    def remove(self):
        """
        Removes the hooks.
        """
        self.removable_handles.remove()

    @abstractmethod
    def register(
        self,
        module_registry: Dict[str, torch.nn.Module],
        args_registry,
    ):
        """
        Registers the hooks.
        """
        pass

    @abstractmethod
    def generate(
        self,
        hook_args,
    ) -> Callable[
        [torch.nn.Module, torch.Tensor, torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Generates a hook.
        """
        pass


class CacheMode(str, Enum):
    """
    Enum for cache mode.
    """

    INPUT = "input"
    OUTPUT = "output"


@dataclass
class CacheHookArgs(metaclass=HookArgs):
    """
    Hook arguments for caching.
    """

    cache: Dict[str, torch.Tensor]
    key: str


class CacheHookFactory(HookFactory):
    """
    Hook factory for caching inputs and outputs.
    """

    def __init__(self, mode: CacheMode):
        """
        Initializes the hook factory.
        """
        super().__init__()
        self._mode = CacheMode(mode)

    @property
    def mode(self):
        """
        Returns the cache mode.
        """
        return self._mode

    def register(
        self,
        module_registry: Dict[str, torch.nn.Module],
        args_registry: Dict[str, CacheHookArgs],
    ):
        """
        Registers the hooks.
        """
        assert set(module_registry.keys()) == set(args_registry.keys())
        for key, module in module_registry.items():
            removable_handle = module.register_forward_hook(
                self.generate(args_registry[key])
            )
            self.removable_handles.append(removable_handle)

    def generate(
        self, hook_args: CacheHookArgs
    ) -> Callable[
        [torch.nn.Module, torch.Tensor, torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Generates a hook.
        """
        if not isinstance(hook_args, CacheHookArgs):
            raise ValueError(f"Invalid hook args type: {type(hook_args)}")
        cache = hook_args.cache
        key = hook_args.key
        if self.mode == CacheMode.INPUT:

            def hook(module, input, output):
                cache[key] = input

        elif self.mode == CacheMode.OUTPUT:

            def hook(module, input, output):
                cache[key] = output

        else:
            raise ValueError(f"Invalid cache mode: {self.mode}")
        return hook
