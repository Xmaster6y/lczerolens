"""
Utils from the lczero executable and bindings.
"""
import subprocess

import chess
import torch
from lczero.backends import Backend, GameState

from lczerolens import move_utils

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


def generic_command(args, verbose=False):
    """
    Run a generic command.
    """
    popen = subprocess.Popen(
        ["lc0", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    popen.wait()
    if popen.returncode != 0:
        if verbose:
            stderr = (
                f'\n[DEBUG] stderr:\n{popen.stderr.read().decode("utf-8")}'
            )
        else:
            stderr = ""
        raise RuntimeError(f"Could not run `lc0 {' '.join(args)}`." + stderr)
    return popen.stdout.read().decode("utf-8")


def describenet(path, verbose=False):
    """
    Describe the net at the given path.
    """
    return generic_command(["describenet", "-w", path], verbose=verbose)


def convert_to_onnx(in_path, out_path, verbose=False):
    """
    Convert the net at the given path.
    """
    return generic_command(
        ["leela2onnx", f"--input={in_path}", f"--output={out_path}"],
        verbose=verbose,
    )


def convert_to_leela(in_path, out_path, verbose=False):
    """
    Convert the net at the given path.
    """
    return generic_command(
        ["onnx2leela", f"--input={in_path}", f"--output={out_path}"],
        verbose=verbose,
    )


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
    only_legal: bool = False,
    illegal_value: float = 0,
):
    """
    Predicts the move.
    """
    filtered_policy = torch.full((1858,), illegal_value, dtype=torch.float)
    lczero_input = lczero_game.as_input(lczero_backend)
    (lczero_output,) = lczero_backend.evaluate(lczero_input)
    if only_legal:
        indices = torch.tensor(lczero_game.policy_indices())
    else:
        indices = torch.tensor(range(1858))
    if softmax:
        policy = torch.tensor(
            lczero_output.p_softmax(*range(1858)), dtype=torch.float
        )
    else:
        policy = torch.tensor(
            lczero_output.p_raw(*range(1858)), dtype=torch.float
        )
    value = torch.tensor(lczero_output.q())
    filtered_policy[indices] = policy[indices]
    return filtered_policy, value


def moves_with_castling_swap(lczero_game: GameState, board: chess.Board):
    """
    Get the moves with castling swap.
    """
    lczero_legal_moves = lczero_game.moves()
    lczero_policy_indices = list(lczero_game.policy_indices())
    for move in board.legal_moves:
        uci_move = move.uci()
        if uci_move in lczero_legal_moves:
            continue
        if board.is_castling(move):
            leela_uci_move = uci_move.replace("g", "h").replace("c", "a")
            if leela_uci_move in lczero_legal_moves:
                lczero_legal_moves.remove(leela_uci_move)
                lczero_legal_moves.append(uci_move)
                lczero_policy_indices.remove(
                    move_utils.encode_move(
                        chess.Move.from_uci(leela_uci_move),
                        (board.turn, not board.turn),
                    )
                )
                lczero_policy_indices.append(
                    move_utils.encode_move(move, (board.turn, not board.turn))
                )
    return lczero_legal_moves, lczero_policy_indices
