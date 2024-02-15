"""Probing lens for XAI.
"""

from typing import Any, Dict

import chess
import torch
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from lczerolens.adapt.wrapper import ModelWrapper
from lczerolens.xai.concept import (
    BinaryConcept,
    Concept,
    ContinuousConcept,
    MulticlassConcept,
    UniqueConceptDataset,
)
from lczerolens.xai.hook import CacheHook, HookConfig
from lczerolens.xai.lens import Lens


class ProbingLens(Lens):
    """
    Class for probing-based XAI methods.
    """

    def __init__(self, concept: Concept):
        if isinstance(concept, BinaryConcept) or isinstance(
            concept, MulticlassConcept
        ):
            self.probe = LogisticRegression(penalty="l1", solver="saga")
        elif isinstance(concept, ContinuousConcept):
            self.probe = Lasso()
        else:
            raise ValueError(f"{concept} is not a valid Concept")
        self.concept = concept

    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        if hasattr(wrapper.model, "block0"):
            return True
        else:
            return False

    def compute_heatmap(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the probing heatmap for a given model and input.
        """
        raise NotImplementedError

    def compute_statistics(
        self,
        dataset: UniqueConceptDataset,
        wrapper: ModelWrapper,
        batch_size: int,
        **kwargs,
    ) -> Dict[str, Dict[Any, Any]]:
        """
        Computes the statistics for a given board.
        """
        if not isinstance(dataset, UniqueConceptDataset):
            raise ValueError(f"{dataset} is not a UniqueConceptDataset")

        test_size = kwargs.get("test_size", 0.2)
        shuffle = kwargs.get("shuffle", True)
        random_state = kwargs.get("random_state", 42)

        cache_hook = CacheHook(HookConfig(module_exp=r"block\d+$"))
        cache_hook.register(wrapper.model)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=UniqueConceptDataset.collate_fn_list,
        )
        labels = []
        activations = {}
        for batch in loader:
            board_list, label_list = batch
            labels.extend(label_list)
            wrapper.predict(board_list)
            for key, value in cache_hook.storage.items():
                if key not in activations:
                    activations[key] = value.flatten(start_dim=1)
                else:
                    activations[key] = torch.cat(
                        [activations[key], value.flatten(start_dim=1)], dim=0
                    )
        labels = torch.Tensor(labels)
        cache_hook.remove()

        statistics: Dict[str, Dict[str, Any]] = {
            "probe_coef": {},
            "metrics": {},
        }
        train_indices, test_indices = train_test_split(
            range(len(labels)),
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        for key, value in activations.items():
            self.probe.fit(value[train_indices], labels[train_indices])
            probe_coef = self.probe.coef_
            predictions = self.probe.predict(value[test_indices])
            metrics = self.concept.compute_metrics(
                predictions, labels[test_indices]
            )
            statistics["probe_coef"][key] = probe_coef
            statistics["metrics"][key] = metrics
        return statistics
