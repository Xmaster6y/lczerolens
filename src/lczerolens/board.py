"""Board class."""

import re
from enum import Enum
from typing import Optional, Generator, Tuple, List, Union, Any

import chess
import chess.svg
import torch
import io
import numpy as np

from .constants import INVERTED_POLICY_INDEX, POLICY_INDEX


class InputEncoding(int, Enum):
    """Input encoding for the board tensor."""

    INPUT_CLASSICAL_112_PLANE = 0
    INPUT_CLASSICAL_112_PLANE_REPEATED = 1
    INPUT_CLASSICAL_112_PLANE_NO_HISTORY = 2


class LczeroBoard(chess.Board):
    """A class for wrapping the LczeroBoard class."""

    @staticmethod
    def get_plane_order(us: bool):
        """Get the plane order for the given us view.

        Parameters
        ----------
        us : bool
            The us_them tuple.

        Returns
        -------
        str
            The plane order.
        """
        plane_orders = {chess.WHITE: "PNBRQK", chess.BLACK: "pnbrqk"}
        return plane_orders[us] + plane_orders[not us]

    @staticmethod
    def get_piece_index(piece: str, us: bool, plane_order: Optional[str] = None):
        """Converts a piece to its index in the plane order.

        Parameters
        ----------
        piece : str
            The piece to convert.
        us : bool
            The us_them tuple.
        plane_order : Optional[str]
            The plane order.

        Returns
        -------
        int
            The index of the piece in the plane order.
        """
        if plane_order is None:
            plane_order = LczeroBoard.get_plane_order(us)
        return f"{plane_order}0".index(piece)

    def to_config_tensor(
        self,
        us: Optional[bool] = None,
        input_encoding: InputEncoding = InputEncoding.INPUT_CLASSICAL_112_PLANE,
    ):
        """Converts a LczeroBoard to a tensor based on the pieces configuration.

        Parameters
        ----------
        us : Optional[bool]
            The us_them tuple.
        input_encoding : InputEncoding
            The input encoding method.

        Returns
        -------
        torch.Tensor
            The 13x8x8 tensor.
        """
        if not isinstance(input_encoding, InputEncoding):
            raise NotImplementedError(f"Input encoding {input_encoding} not implemented.")
        if us is None:
            us = self.turn
        plane_order = LczeroBoard.get_plane_order(us)

        def piece_to_index(piece: str):
            return f"{plane_order}0".index(piece)

        fen_board = self.fen().split(" ")[0]
        fen_rep = re.sub(r"(\d)", lambda m: "0" * int(m.group(1)), fen_board)
        rows = fen_rep.split("/")
        rev_rows = rows[::-1]
        ordered_fen = "".join(rev_rows)

        config_tensor = torch.zeros((13, 8, 8), dtype=torch.float)
        ordinal_board = torch.tensor(tuple(map(piece_to_index, ordered_fen)), dtype=torch.float)
        ordinal_board = ordinal_board.reshape((8, 8)).unsqueeze(0)
        piece_tensor = torch.tensor(tuple(map(piece_to_index, plane_order)), dtype=torch.float)
        piece_tensor = piece_tensor.reshape((12, 1, 1))
        config_tensor[:12] = (ordinal_board == piece_tensor).float()
        if self.is_repetition(2):  # Might be wrong if the full history is not available
            config_tensor[12] = torch.ones((8, 8), dtype=torch.float)
        return config_tensor if us == chess.WHITE else config_tensor.flip(1)

    def to_input_tensor(
        self,
        with_history: bool = True,
        input_encoding: InputEncoding = InputEncoding.INPUT_CLASSICAL_112_PLANE,
    ):
        """Create the lc0 input tensor from the history of a game.

        Parameters
        ----------
        with_history : bool
            Whether to include the history of the game.
        input_encoding : InputEncoding
            The input encoding method.

        Returns
        -------
        torch.Tensor
            The 112x8x8 tensor.
        """
        if not isinstance(input_encoding, InputEncoding):
            raise NotImplementedError(f"Input encoding {input_encoding} not implemented.")

        input_tensor = torch.zeros((112, 8, 8), dtype=torch.float)
        us = self.turn
        them = not us
        moves = []

        if with_history:
            if (
                input_encoding == InputEncoding.INPUT_CLASSICAL_112_PLANE
                or input_encoding == InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED
            ):
                for i in range(8):
                    config_tensor = self.to_config_tensor(us)
                    input_tensor[i * 13 : (i + 1) * 13] = config_tensor
                    try:
                        moves.append(self.pop())
                    except IndexError:
                        if input_encoding == InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED:
                            input_tensor[(i + 1) * 13 : 104] = config_tensor.repeat(7 - i, 1, 1)
                        break

            elif input_encoding == InputEncoding.INPUT_CLASSICAL_112_PLANE_NO_HISTORY:
                config_tensor = self.to_config_tensor(us)
                input_tensor[:104] = config_tensor.repeat(8, 1, 1)
            else:
                raise ValueError(f"Got unexpected input encoding {input_encoding}")

        # Restore the moves
        for move in reversed(moves):
            self.push(move)

        if self.has_queenside_castling_rights(us):
            input_tensor[104] = torch.ones((8, 8), dtype=torch.float)
        if self.has_kingside_castling_rights(us):
            input_tensor[105] = torch.ones((8, 8), dtype=torch.float)
        if self.has_queenside_castling_rights(them):
            input_tensor[106] = torch.ones((8, 8), dtype=torch.float)
        if self.has_kingside_castling_rights(them):
            input_tensor[107] = torch.ones((8, 8), dtype=torch.float)
        if us == chess.BLACK:
            input_tensor[108] = torch.ones((8, 8), dtype=torch.float)
        input_tensor[109] = torch.ones((8, 8), dtype=torch.float) * self.halfmove_clock
        input_tensor[111] = torch.ones((8, 8), dtype=torch.float)

        return input_tensor

    @staticmethod
    def encode_move(
        move: chess.Move,
        us: bool,
    ) -> int:
        """
        Converts a chess.Move object to an index.

        Parameters
        ----------
        move : chess.Move
            The chess move to encode.
        us : bool
            The side to move (True for white, False for black).

        Returns
        -------
        int
            The encoded move index.
        """
        from_square = move.from_square
        to_square = move.to_square

        if us == chess.BLACK:
            from_square_row = from_square // 8
            from_square_col = from_square % 8
            from_square = 8 * (7 - from_square_row) + from_square_col
            to_square_row = to_square // 8
            to_square_col = to_square % 8
            to_square = 8 * (7 - to_square_row) + to_square_col
        us_uci_move = chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]
        if move.promotion is not None:
            if move.promotion == chess.BISHOP:
                us_uci_move += "b"
            elif move.promotion == chess.ROOK:
                us_uci_move += "r"
            elif move.promotion == chess.QUEEN:
                us_uci_move += "q"
            # Knight promotion is the default
        return INVERTED_POLICY_INDEX[us_uci_move]

    def decode_move(
        self,
        index: int,
    ) -> chess.Move:
        """
        Converts an index to a chess.Move object.

        Parameters
        ----------
        index : int
            The index to convert.

        Returns
        -------
        chess.Move
            The chess move.
        """
        us = self.turn
        us_uci_move = POLICY_INDEX[index]
        from_square = chess.SQUARE_NAMES.index(us_uci_move[:2])
        to_square = chess.SQUARE_NAMES.index(us_uci_move[2:4])
        if us == chess.BLACK:
            from_square_row = from_square // 8
            from_square_col = from_square % 8
            from_square = 8 * (7 - from_square_row) + from_square_col
            to_square_row = to_square // 8
            to_square_col = to_square % 8
            to_square = 8 * (7 - to_square_row) + to_square_col

        uci_move = chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]
        from_piece = self.piece_at(from_square)
        if from_piece == chess.PAWN and to_square >= 56:  # Knight promotion is the default
            uci_move += "n"
        return chess.Move.from_uci(uci_move)

    def get_legal_indices(
        self,
    ) -> torch.Tensor:
        """
        Gets the legal indices.

        Returns
        -------
        torch.Tensor
            Tensor containing indices of legal moves.
        """
        us = self.turn
        return torch.tensor([self.encode_move(move, us) for move in self.legal_moves])

    def get_next_legal_boards(
        self,
        n_history: int = 7,
    ) -> Generator["LczeroBoard", None, None]:
        """
        Gets the next legal boards.

        Parameters
        ----------
        n_history : int, optional
            Number of previous positions to keep in the move stack, by default 7.

        Returns
        -------
        Generator[LczeroBoard, None, None]
            Generator yielding board positions after each legal move.
        """
        working_board = self.copy(stack=n_history)
        for move in working_board.legal_moves:
            working_board.push(move)
            yield working_board.copy(stack=n_history)
            working_board.pop()

    def render_heatmap(
        self,
        heatmap: Union[torch.Tensor, np.ndarray],
        square: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        arrows: Optional[List[Tuple[str, str]]] = None,
        normalise: str = "none",
        save_to: Optional[str] = None,
        cmap_name: str = "RdYlBu_r",
        alpha: float = 1.0,
    ) -> Tuple[Optional[str], Any]:
        """Render a heatmap on the board.

        Parameters
        ----------
        heatmap : torch.Tensor or numpy.ndarray
            The heatmap values to visualize on the board (64,).
        square : Optional[str], default=None
            Chess square to highlight (e.g. 'e4').
        vmin : Optional[float], default=None
            Minimum value for the colormap normalization.
        vmax : Optional[float], default=None
            Maximum value for the colormap normalization.
        arrows : Optional[List[Tuple[str, str]]], default=None
            List of arrow tuples (from_square, to_square) to draw on board.
        normalise : str, default="none"
            Normalization method. Use "abs" for absolute value normalization.
        save_to : Optional[str], default=None
            Path to save the visualization. If None, returns the figure.
        cmap_name : str, default="RdYlBu_r"
            Name of matplotlib colormap to use.
        alpha : float, default=1.0
            Opacity of the heatmap overlay.

        Returns
        -------
        Union[Tuple[str, matplotlib.figure.Figure], None]
            If save_to is None, returns (SVG string, matplotlib figure).
            If save_to is provided, saves files and returns None.

        Raises
        ------
        ValueError
            If save_to is provided and does not end with `.svg`.
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required to render heatmaps, install it with `pip install lczerolens[viz]`."
            ) from e

        cmap = matplotlib.colormaps[cmap_name].resampled(1000)

        if normalise == "abs":
            a_max = heatmap.abs().max()
            if a_max != 0:
                heatmap = heatmap / a_max
            vmin = -1
            vmax = 1
        if vmin is None:
            vmin = heatmap.min()
        if vmax is None:
            vmax = heatmap.max()
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)

        color_dict = {}
        for square_index in range(64):
            color = cmap(norm(heatmap[square_index]))
            color = (*color[:3], alpha)
            color_dict[square_index] = matplotlib.colors.to_hex(color, keep_alpha=True)
        fig = plt.figure(figsize=(1, 4.1))
        ax = plt.gca()
        ax.axis("off")
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation="vertical",
            fraction=1.0,
        )
        if square is not None:
            try:
                check = chess.parse_square(square)
            except ValueError:
                check = None
        else:
            check = None
        if arrows is None:
            arrows = []

        svg_board = chess.svg.board(
            self,
            check=check,
            fill=color_dict,
            size=400,
            arrows=arrows,
        )
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg")
        svg_colorbar = buffer.getvalue().decode("utf-8")
        plt.close()

        if save_to is None:
            return svg_board, svg_colorbar
        elif not save_to.endswith(".svg"):
            raise ValueError("only saving to `svg` is supported")

        with open(save_to.replace(".svg", "_board.svg"), "w") as f:
            f.write(svg_board)
        with open(save_to.replace(".svg", "_colorbar.svg"), "w") as f:
            f.write(svg_colorbar)
