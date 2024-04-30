"""Probing lens for XAI."""

from typing import Any, Callable, Dict, Optional

import chess
import torch
from torch.utils.data import Dataset

from lczerolens.model.wrapper import ModelWrapper
from lczerolens.xai.hook import HookConfig, MeasureHook
from lczerolens.xai.lens import Lens
from lczerolens.xai.probe import Probe


@Lens.register("probing")
class ProbingLens(Lens):
    """
    Class for probing-based XAI methods.
    """

    def __init__(self, probe_dict: Dict[str, Probe]):
        self.probe_dict = probe_dict
        self.measure_hooks = {}
        for module_name, probe in self.probe_dict.items():
            self.measure_hooks[module_name] = MeasureHook(HookConfig(module_exp=module_name, data_fn=probe.predict))

    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        return isinstance(wrapper.model, torch.nn.Module)

    def analyse_board(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> torch.Tensor:
        """
        Analyse a single board with the probing lens.
        """
        measures = {}
        for measure_hook in self.measure_hooks.values():
            measure_hook.clear()
            measure_hook.register(wrapper.model)
        wrapper.predict(board)
        for measure_hook in self.measure_hooks.values():
            measures.update(measure_hook.storage)
            measure_hook.clear()
        return measures

    def analyse_dataset(
        self,
        dataset: Dataset,
        wrapper: ModelWrapper,
        batch_size: int,
        collate_fn: Optional[Callable] = None,
        save_to: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Analyse a dataset with the probing lens.
        """
        if save_to is not None:
            raise NotImplementedError("Saving to file is not implemented.")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        batched_measures: Dict[str, Any] = {}
        for measure_hook in self.measure_hooks.values():
            measure_hook.clear()
            measure_hook.register(wrapper.model)
        for batch in dataloader:
            indices, boards = batch
            wrapper.predict(boards)
            for module_name, measure_hook in self.measure_hooks.items():
                if module_name not in batched_measures:
                    batched_measures[module_name] = measure_hook.storage[module_name]
                else:
                    batched_measures[module_name] = torch.cat(
                        (
                            batched_measures[module_name],
                            measure_hook.storage[module_name],
                        ),
                        dim=0,
                    )
        for measure_hook in self.measure_hooks.values():
            measure_hook.clear()
        return batched_measures
