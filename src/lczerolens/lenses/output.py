"""Output lens."""

from lczerolens.model import LczeroModel
from lczerolens.lens import Lens


@Lens.register("output")
class OutputLens(Lens):
    """
    Class for output-based XAI methods.

    Examples
    --------

        .. code-block:: python

            model = LczeroModel.from_path(model_path)
            lens = OutputLens()
            board = LczeroBoard()
            results = lens.analyse(board, model=model)
    """

    def _intervene(
        self,
        model: LczeroModel,
        **kwargs,
    ) -> dict:
        output = model.output.save()
        return {"output": output}
