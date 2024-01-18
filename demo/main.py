"""
Gradio demo for lczero-easy.
"""

import gradio as gr

demo = gr.TabbedInterface(
    [],
    [],
    title="Lc0 Easy",
    analytics_enabled=False,
)

if __name__ == "__main__":
    demo.launch(
        server_port=8000,
        server_name="0.0.0.0",
    )
