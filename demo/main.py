"""
Gradio demo for lczero-easy.
"""

import gradio as gr

from . import convert_interface

demo = gr.TabbedInterface(
    [convert_interface.interface],
    ["Convert"],
    title="LczeroLens Demo",
    analytics_enabled=False,
)

if __name__ == "__main__":
    demo.launch(
        server_port=8000,
        server_name="0.0.0.0",
    )
