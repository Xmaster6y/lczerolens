"""Utils from the lczero executable and bindings.

Notes
----
The lczero bindings are not installed by default. You can install them by
running `pip install lczerolens[backends]`.
"""

import subprocess

import chess
import torch

from lczerolens._codec import encode_move

try:
    from lczero.backends import Backend, GameState
except ImportError as e:
    raise ImportError(
        "LCZero bindings are required to use the backends, install them with `pip install lczerolens[backends]`."
    ) from e


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
            stderr = f"\n[DEBUG] stderr:\n{popen.stderr.read().decode('utf-8')}"
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


def board_from_backend(lczero_backend: Backend, lczero_game: GameState, planes: int = 112):
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
        policy = torch.tensor(lczero_output.p_softmax(*range(1858)), dtype=torch.float)
    else:
        policy = torch.tensor(lczero_output.p_raw(*range(1858)), dtype=torch.float)
    value = torch.tensor(lczero_output.q())
    filtered_policy[indices] = policy[indices]
    return filtered_policy, value


def moves_with_castling_swap(lczero_game: GameState, board: chess.Board):
    """
    Return backend legal moves in the project's canonical castling notation.

    lc0 has historically represented castling in the policy head as a king
    move to the rook square (``e1h1`` / ``e1a1``).  Bindings have used both
    that notation and standard UCI (``e1g1`` / ``e1c1``) for ``moves()``,
    while retaining the rook-square policy index.  Normalize both forms to
    the fixed Lczero policy vocabulary.
    """
    lczero_legal_moves = list(lczero_game.moves())
    lczero_policy_indices = list(lczero_game.policy_indices())
    for move in board.legal_moves:
        if not board.is_castling(move):
            continue

        uci_move = move.uci()
        rook_file = 7 if board.is_kingside_castling(move) else 0
        rook_square = chess.square(rook_file, chess.square_rank(move.from_square))
        leela_uci_move = chess.Move(move.from_square, rook_square).uci()
        try:
            index = next(
                i for i, backend_move in enumerate(lczero_legal_moves) if backend_move in (uci_move, leela_uci_move)
            )
        except StopIteration:
            continue

        lczero_legal_moves[index] = uci_move
        lczero_policy_indices[index] = encode_move(board, move)
    return lczero_legal_moves, lczero_policy_indices
