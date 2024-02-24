"""Nice plotting of chessboard and heatmap with arrows.
"""


import chess
from pylatex import Figure, NoEscape, SubFigure


def add_plot(
    doc,
    label,
    heatmap_str_list,
    current_piece_pos=None,
    next_move=None,
    caption=None,
    heatmap_caption_list=None,
):
    # Put some data inside the Figure environment
    with doc.create(Figure()) as fig:
        doc.append(NoEscape(r"\centering"))
        if caption is not None:
            fig.add_caption(caption)
        verbatim = NoEscape(
            r"\storechessboardstyle{8x8}{maxfield=h8,showmover=true}"
        )
        doc.append(verbatim)

        with doc.create(
            SubFigure(
                width=NoEscape(r"0.45\textwidth"),
            )
        ) as subfig:
            subfig.add_caption("Board")
            doc.append(NoEscape(r"\chessboard[style=8x8,"))
            if current_piece_pos is not None:
                markmove = current_piece_pos + "-" + next_move
                markfields = (
                    "{{" + current_piece_pos + "},{" + next_move + "}}"
                )
                chessboard_fen = NoEscape(
                    rf"setfen={label},showmover=true,"
                    rf"color=green,pgfstyle=straightmove,markmove={markmove},"
                    rf"pgfstyle=border,color=red,markfields={markfields},]"
                )
            else:
                chessboard_fen = NoEscape(
                    rf"\chessboard[style=8x8,setfen={label},"
                    "showmover=true,pgfstyle=straightmove,color=green,]"
                )
            doc.append(chessboard_fen)
        for i, heatmap_str in enumerate(heatmap_str_list):
            doc.append(NoEscape(r"\hfill"))
            with doc.create(
                SubFigure(width=NoEscape(r"0.45\textwidth"))
            ) as subfig:
                subfig.add_caption(heatmap_caption_list[i])
                heatmap_begin = NoEscape(
                    r"\chessboard[style=8x8,showmover=false,"
                )
                doc.append(heatmap_begin)

                heatmap_end = NoEscape(heatmap_str) + NoEscape(r"]")
                doc.append(heatmap_end)
    return doc


def create_heatmap_string(heatmap, abs_max=True):
    if abs_max:
        heatmap = heatmap / heatmap.abs().max()
    heatmap_str = ""
    for idx, name in enumerate(chess.SQUARE_NAMES):
        colorcode = heatmap[idx]
        if colorcode >= 0:
            heatmap_str += (
                "pgfstyle=color, color=red!"
                f"{colorcode*100:.0f}!white, markfield={name},\n"
            )
        elif colorcode < 0:
            heatmap_str += (
                "pgfstyle=color, color=blue!"
                f"{-colorcode*100:.0f}!white, markfield={name},\n"
            )
        else:
            raise TypeError
    return heatmap_str
