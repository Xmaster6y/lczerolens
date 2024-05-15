"""Probing lens for XAI."""

from typing import Dict, Iterator

import chess
import torch

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

    def analyse_batched_boards(
        self,
        iter_boards: Iterator,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> Iterator:
        """Computes the statistics for a given board.

        Parameters
        ----------
        iter_boards : Iterator
            The iterator over the boards.
        wrapper : ModelWrapper
            The model wrapper.

        Returns
        -------
        Iterator
            The iterator over the statistics.
        """
        for measure_hook in self.measure_hooks.values():
            measure_hook.clear()
            measure_hook.register(wrapper.model)
        for batch in iter_boards:
            boards, *_ = batch
            wrapper.predict(boards)
            merged_measures = {}
            for module_name, measure_hook in self.measure_hooks.items():
                merged_measures[module_name] = measure_hook.storage[module_name]
            yield merged_measures
        for measure_hook in self.measure_hooks.values():
            measure_hook.clear()
