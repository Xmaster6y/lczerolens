"""
Gradio demo for lczero-easy.
"""

import gradio as gr

from . import (
    attention_interface,
    backend_interface,
    board_interface,
    convert_interface,
    encoding_interface,
    policy_interface,
)

demo = gr.TabbedInterface(
    [
        attention_interface.interface,
        policy_interface.interface,
        backend_interface.interface,
        encoding_interface.interface,
        board_interface.interface,
        convert_interface.interface,
    ],
    ["Attention", "Policy", "Backend", "Encoding", "Board", "Convert"],
    title="LczeroLens Demo",
    analytics_enabled=False,
)

if __name__ == "__main__":
    demo.launch(
        server_port=8000,
        server_name="0.0.0.0",
    )
