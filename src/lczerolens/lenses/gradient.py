"""Compute Gradient heatmap for a given model and input."""

from lczerolens.model import LczeroModel
from lczerolens.lens import Lens


@Lens.register("gradient")
class GradientLens(Lens):
    """Class for gradient-based XAI methods."""

    def __init__(self, *, input_requires_grad: bool = True, **kwargs):
        self.input_requires_grad = input_requires_grad
        super().__init__(**kwargs)

    def _intervene(self, model: LczeroModel, **kwargs) -> dict:
        init_target = kwargs.get("init_target", lambda model: model.output["value"])
        init_gradient = kwargs.get("init_gradient", lambda model: None)

        results = {}
        if self.input_requires_grad:
            model.input.requires_grad_(self.input_requires_grad)
            results["input_grad"] = model.input.grad.save()
        for name, module in self._get_modules(model):
            results[f"{name}_output_grad"] = module.output.grad.save()
        target = init_target(model)
        target.backward(gradient=init_gradient(model))
        return results
