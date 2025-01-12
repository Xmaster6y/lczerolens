"""Compute Gradient heatmap for a given model and input."""

from lczerolens.model import LczeroModel
from lczerolens.lens import Lens


@Lens.register("gradient")
class GradientLens(Lens):
    """Class for gradient-based XAI methods."""

    def _intervene(self, model: LczeroModel, **kwargs) -> dict:
        target = kwargs.get("target", "policy")
        init_fn = kwargs.get("init_fn", lambda output, **kwargs: output)

        model.input.requires_grad_(True)
        gradient = model.input.grad.save()
        output = model.output[target] if target is not None else model.output
        output.backward(gradient=init_fn(output, **kwargs))
        return {"gradient": gradient}
