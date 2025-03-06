"""Activation lens."""

from lczerolens.model import LczeroModel
from lczerolens.lens import Lens


@Lens.register("activation")
class ActivationLens(Lens):
    """
    Class for activation-based XAI methods.

    Examples
    --------

        .. code-block:: python

            model = LczeroModel.from_path(model_path)
            lens = ActivationLens()
            board = LczeroBoard()
            results = lens.analyse(board, model=model)
    """

    def _intervene(
        self,
        model: LczeroModel,
        **kwargs,
    ) -> dict:
        save_inputs = kwargs.get("save_inputs", False)
        results = {}
        for name, module in self._get_modules(model):
            if save_inputs:
                results[f"{name}_input"] = module.input.save()
            results[f"{name}_output"] = module.output.save()
        return results
