"""Helpers to modify the default classes."""

import warnings
from collections.abc import Iterable
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from crp.helper import load_maximization, load_statistics
from crp.hooks import FeatVisHook
from crp.image import vis_img_heatmap
from crp.visualization import FeatureVisualization
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from zennit.composites import Composite, NameMapComposite


class DynamicSampler(Sampler[int]):
    """
    Samples elements from a given list of indices.
    """

    def __init__(self, indices: Optional[List[int]] = None) -> None:
        if indices is None:
            self.indices = []
        else:
            self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def set_indices(self, indices):
        self.indices = indices


class ModifiedFeatureVisualization(FeatureVisualization):
    def collate_fn(self, batch, preprocessing=True):
        data, targets = zip(*batch)
        data_tensor = torch.stack(data, dim=0)
        if preprocessing:
            data_tensor = self.preprocess_data(data_tensor)
        data_tensor.requires_grad = True
        return data_tensor, targets

    def run(
        self,
        composite: Composite,
        batch_size=32,
        checkpoint=500,
        on_device=None,
        custom_collate_fn=None,
        num_workers=4,
    ):
        print("Running Analysis...")
        saved_checkpoints = self.run_distributed(
            composite,
            batch_size,
            checkpoint,
            on_device,
            custom_collate_fn,
            num_workers,
        )

        print("Collecting results...")
        saved_files = self.collect_results(saved_checkpoints)

        return saved_files

    def run_distributed(
        self,
        composite: Composite,
        batch_size=16,
        checkpoint=500,
        on_device=None,
        custom_collate_fn=None,
        num_workers=4,
    ):
        """
        max batch_size = max(multi_targets) * data_batch
        data_end: exclusively counted
        """

        self.saved_checkpoints = {
            "r_max": [],
            "a_max": [],
            "r_stats": [],
            "a_stats": [],
        }  # type: ignore
        last_checkpoint = 0
        samples = np.arange(len(self.dataset))

        name_map, dict_inputs = [], {}  # type: ignore
        for l_name, concept in self.layer_map.items():
            hook = FeatVisHook(self, concept, l_name, dict_inputs, on_device)
            name_map.append(([l_name], hook))
        fv_composite = NameMapComposite(name_map)

        if composite:
            composite.register(self.attribution.model)
        fv_composite.register(self.attribution.model)

        if custom_collate_fn is None:

            def collate_fn(batch):
                return self.collate_fn(batch, preprocessing=True)

        else:
            collate_fn = custom_collate_fn
        loader = DataLoader(
            self.dataset,
            sampler=DynamicSampler(samples),
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        for b, batch in tqdm(
            enumerate(loader),
            total=len(self.dataset) // batch_size,
            dynamic_ncols=True,
        ):
            samples_batch = samples[b * batch_size : (b + 1) * batch_size]
            data_batch, targets_samples = batch

            targets_samples = np.array(targets_samples)  # numpy operation needed

            # convert multi target to single target if user defined the method
            data_broadcast, targets, sample_indices = [], [], []
            try:
                for i_t, target in enumerate(targets_samples):
                    single_targets = self.multitarget_to_single(target)
                    for st in single_targets:
                        targets.append(st)
                        data_broadcast.append(data_batch[i_t])
                        sample_indices.append(samples_batch[i_t])
                if len(data_broadcast) == 0:
                    continue
                # TODO: test stack
                data_broadcast = torch.stack(data_broadcast, dim=0)
                sample_indices = np.array(sample_indices)
                targets = np.array(targets)

            except NotImplementedError:
                data_broadcast, targets, sample_indices = (
                    data_batch,
                    targets_samples,
                    samples_batch,
                )

            conditions = [{self.attribution.MODEL_OUTPUT_NAME: [t]} for t in targets]
            # dict_inputs is linked to FeatHooks
            dict_inputs["sample_indices"] = sample_indices
            dict_inputs["targets"] = targets

            # composites are already registered before
            if on_device:
                data_broadcast = data_broadcast.to(on_device)  # type: ignore
            self.attribution(data_broadcast, conditions, None, exclude_parallel=False)

            if b % checkpoint == checkpoint - 1:
                self._save_results((last_checkpoint, sample_indices[-1] + 1))
                last_checkpoint = sample_indices[-1] + 1

        # TODO: what happens if result arrays are empty?
        self._save_results((last_checkpoint, sample_indices[-1] + 1))

        if composite:
            composite.remove()
        fv_composite.remove()

        return self.saved_checkpoints

    @FeatureVisualization.cache_reference
    def get_max_reference(
        self,
        concept_ids: Union[int, list],
        layer_name: str,
        mode="relevance",
        r_range: Tuple[int, int] = (0, 8),
        composite: Composite = None,
        rf=False,
        plot_fn=vis_img_heatmap,
        batch_size=32,
        custom_collate_fn=None,
    ) -> Dict:
        ref_c = {}
        if not isinstance(concept_ids, Iterable):
            concept_ids = [concept_ids]
        if mode == "relevance":
            d_c_sorted, _, rf_c_sorted = load_maximization(self.RelMax.PATH, layer_name)
        elif mode == "activation":
            d_c_sorted, _, rf_c_sorted = load_maximization(self.ActMax.PATH, layer_name)
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")

        if rf and not composite:
            warnings.warn(
                "The receptive field is only computed, if you fill the "
                "'composite' argument with a zennit Composite."
            )
        for c_id in concept_ids:
            d_indices = d_c_sorted[r_range[0] : r_range[1], c_id]
            n_indices = rf_c_sorted[r_range[0] : r_range[1], c_id]

            ref_c[c_id] = self._load_ref_and_attribution(
                d_indices,
                c_id,
                n_indices,
                layer_name,
                composite,
                rf,
                plot_fn,
                batch_size,
                custom_collate_fn,
            )
        return ref_c

    @FeatureVisualization.cache_reference
    def get_stats_reference(
        self,
        concept_id: int,
        layer_name: str,
        targets: Union[int, list],
        mode="relevance",
        r_range: Tuple[int, int] = (0, 8),
        composite=None,
        rf=False,
        plot_fn=vis_img_heatmap,
        batch_size=32,
        custom_collate_fn=None,
    ):
        ref_t = {}
        if not isinstance(targets, Iterable):
            targets = [targets]
        if mode == "relevance":
            path = self.RelStats.PATH
        elif mode == "activation":
            path = self.ActStats.PATH
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")

        if rf and not composite:
            warnings.warn(
                "The receptive field is only computed, if you fill the"
                " 'composite' argument with a zennit Composite."
            )

        for t in targets:
            d_c_sorted, _, rf_c_sorted = load_statistics(path, layer_name, t)
            d_indices = d_c_sorted[r_range[0] : r_range[1], concept_id]
            n_indices = rf_c_sorted[r_range[0] : r_range[1], concept_id]

            ref_t[f"{concept_id}:{t}"] = self._load_ref_and_attribution(
                d_indices,
                concept_id,
                n_indices,
                layer_name,
                composite,
                rf,
                plot_fn,
                batch_size,
                custom_collate_fn,
            )

        return ref_t

    def _load_ref_and_attribution(
        self,
        d_indices,
        c_id,
        n_indices,
        layer_name,
        composite,
        rf,
        plot_fn,
        batch_size,
        custom_collate_fn,
    ):
        if custom_collate_fn is None:

            def collate_fn(batch):
                return self.collate_fn(batch, preprocessing=False)

        else:
            collate_fn = custom_collate_fn
        loader = DataLoader(
            self.dataset,
            sampler=DynamicSampler(d_indices),
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=1,
        )
        data_batch_list, _ = zip(*loader)

        if composite:
            data_batch = torch.cat(data_batch_list, dim=0)
            data_p = self.preprocess_data(data_batch)
            heatmaps = self._attribution_on_reference(data_p, c_id, layer_name, composite, rf, n_indices, batch_size)

            if callable(plot_fn):
                return plot_fn(data_batch.detach(), heatmaps.detach(), rf)
            else:
                return data_batch.detach().cpu(), heatmaps.detach().cpu()

        else:
            return data_batch_list
