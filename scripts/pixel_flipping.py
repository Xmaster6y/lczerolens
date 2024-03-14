"""Script to perform pixel flipping.

Run with:
```bash
poetry run python -m scripts.pixel_flipping
```
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from crp.attribution import CondAttribution

from lczerolens.game import PolicyFlow
from lczerolens.xai import ConceptDataset, LrpLens
from lczerolens.xai.hook import HookConfig, ModifyHook

#######################################
# HYPERPARAMETERS
#######################################
debug = False
model_name = "64x6-2018_0627_1913_08_161.onnx"
dataset_name = "TCEC_game_collection_random_boards_bestlegal.jsonl"
n_samples = 10
n_steps = 100
viz_name = "pixel_flipping_tcec_bestlegal"
#######################################


model = PolicyFlow.from_path(f"./assets/{model_name}")
concept_dataset = ConceptDataset(f"./assets/{dataset_name}", first_n=n_samples)
print(f"[INFO] Board dataset len: {len(concept_dataset)}")


layer_names = ["model.inputconv", "model.block0/conv2/relu"]
print(layer_names)
dataloader = torch.utils.data.DataLoader(
    concept_dataset,
    batch_size=n_samples,
    shuffle=False,
    collate_fn=ConceptDataset.collate_fn_tensor,
)
indices, board_tensor, labels = next(iter(dataloader))
rule_names = ["default", "no_onnx"]

morf_logit_dict = {
    rule_name: {layer_name: [] for layer_name in layer_names}
    for rule_name in rule_names
}
lerf_logit_dict = {
    rule_name: {layer_name: [] for layer_name in layer_names}
    for rule_name in rule_names
}


def mask_fn(output, modify_data):
    if modify_data is None:
        return output
    else:
        return output * modify_data


for logit_dict, morf in zip([morf_logit_dict, lerf_logit_dict], [True, False]):
    for rule_name in rule_names:
        if rule_name == "default":
            composite = LrpLens.make_default_composite()
            replace_onnx2torch = True
        elif rule_name == "no_onnx":
            composite = LrpLens.make_default_composite()
            replace_onnx2torch = False
        else:
            raise ValueError(f"Unknown rule: {rule_name}")
        for layer_name in layer_names:
            hook_config = HookConfig(
                module_exp=rf"^{layer_name}$",
                data={layer_name: None},
                data_fn=mask_fn,
            )
            hook = ModifyHook(hook_config)
            hook.register(model)
            for i in range(n_steps):
                label_tensor = torch.tensor(labels)

                def init_rel_fn(out_tensor):
                    rel = torch.zeros_like(out_tensor)
                    for i in range(rel.shape[0]):
                        rel[i, label_tensor[i]] = out_tensor[
                            i, label_tensor[i]
                        ]
                    return rel

                board_tensor.requires_grad = True
                with LrpLens.context(
                    model,
                    composite=composite,
                    replace_onnx2torch=replace_onnx2torch,
                ) as modifed_model:
                    attribution = CondAttribution(modifed_model)
                    attr = attribution(
                        board_tensor,
                        [{"y": None}],
                        composite,
                        record_layer=layer_names,
                        init_rel=init_rel_fn,
                    )
                    latent_rel = attr.relevances[layer_name]
                    if morf:
                        to_flip = latent_rel.view(
                            board_tensor.shape[0], -1
                        ).argmax(dim=1)
                    else:
                        to_flip = latent_rel.view(
                            board_tensor.shape[0], -1
                        ).argmin(dim=1)
                    if hook.config.data[layer_name] is None:
                        mask_flat = torch.ones_like(latent_rel).view(
                            board_tensor.shape[0], -1
                        )
                        for i in range(mask_flat.shape[0]):
                            mask_flat[i, to_flip[i]] = 0
                        hook.config.data[layer_name] = mask_flat.view_as(
                            latent_rel
                        )
                    else:
                        old_mask_flat = hook.config.data[layer_name].view(
                            board_tensor.shape[0], -1
                        )
                        for i in range(old_mask_flat.shape[0]):
                            old_mask_flat[i, to_flip[i]] = 0
                        hook.config.data[layer_name] = old_mask_flat.view_as(
                            latent_rel
                        )
                    if debug:
                        print(f"[INFO] Most relevant pixels: {to_flip}")
                    logit_dict[rule_name][layer_name].append(
                        attr.prediction.gather(
                            1, label_tensor.view(-1, 1)
                        ).detach()
                    )
            print(f"[INFO] Layer: {layer_name} done")
            if debug:
                print(
                    "[INFO] Logits: "
                    f"{torch.cat(logit_dict[rule_name][layer_name], dim=1)}"
                )
            hook.remove()
            hook.clear()
        print(f"[INFO] Rule: {rule_name} done")

fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
for rule_name in rule_names:
    for layer_name in layer_names:
        morf_logits = torch.cat(morf_logit_dict[rule_name][layer_name], dim=1)
        lerf_logits = torch.cat(lerf_logit_dict[rule_name][layer_name], dim=1)
        diff = lerf_logits - morf_logits
        means = diff.mean(dim=0)
        stds = diff.std(dim=0)
        ax[0].errorbar(
            np.arange(means.shape[0]),
            means,
            yerr=stds,
            label=f"{rule_name} {layer_name}",
        )
        means = morf_logits.mean(dim=0)
        stds = morf_logits.std(dim=0)
        ax[1].errorbar(
            np.arange(means.shape[0]),
            means,
            yerr=stds,
            label=f"{rule_name} {layer_name}",
        )
        means = lerf_logits.mean(dim=0)
        stds = lerf_logits.std(dim=0)
        ax[2].errorbar(
            np.arange(means.shape[0]),
            means,
            yerr=stds,
            label=f"{rule_name} {layer_name}",
        )
plt.sca(ax[0])
plt.ylabel(f"Mean logit (n={n_samples})")
plt.legend()
plt.sca(ax[1])
plt.xlabel("Pixels flipped", loc="center")
plt.legend()
plt.sca(ax[2])
plt.legend()

plt.savefig(f"./scripts/results/{viz_name}.png")
