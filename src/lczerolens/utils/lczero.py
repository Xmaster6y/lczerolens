"""
Utils from the lczero executable and bindings.
"""
import subprocess

import torch
from lczero.backends import Backend, GameState

try:
    subprocess.run(
        ["lc0", "--help"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
except subprocess.CalledProcessError:
    raise ImportError(
        "LCZero is not installed. Please install it from the sources"
    )


def describenet(path):
    """
    Describe the net at the given path.
    """
    popen = subprocess.Popen(
        ["lc0", "describenet", "-w", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    popen.wait()
    if popen.returncode != 0:
        raise ValueError(f"Could not describe net at {path}.")
    return popen.stdout.read().decode("utf-8")


def convertnet(in_path, out_path):
    """
    Convert the net at the given path.
    """
    popen = subprocess.Popen(
        ["lc0", "leela2onnx", f"--input={in_path}", f"--output={out_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    popen.wait()
    if popen.returncode != 0:
        raise ValueError(f"Could not convert net at {in_path}.")
    return popen.stdout.read().decode("utf-8")


def board_from_backend(
    lczero_backend: Backend, lczero_game: GameState, planes: int = 112
):
    """
    Create a board from the lczero backend.
    """
    lczero_input = lczero_game.as_input(lczero_backend)
    lczero_input_tensor = torch.zeros((112, 64), dtype=torch.float)
    for plane in range(planes):
        mask_str = f"{lczero_input.mask(plane):b}".zfill(64)
        lczero_input_tensor[plane] = torch.tensor(
            tuple(map(int, reversed(mask_str))), dtype=torch.float
        ) * lczero_input.val(plane)
    return lczero_input_tensor.view((112, 8, 8))


def prediction_from_backend(
    lczero_backend: Backend,
    lczero_game: GameState,
    softmax: bool = False,
):
    """
    Predicts the move.
    """
    lczero_input = lczero_game.as_input(lczero_backend)
    (lczero_output,) = lczero_backend.evaluate(lczero_input)
    if softmax:
        policy = lczero_output.p_softmax(*range(1858))
    else:
        policy = lczero_output.p_raw(*range(1858))
    value = lczero_output.q()
    return torch.tensor(policy), torch.tensor(value)
