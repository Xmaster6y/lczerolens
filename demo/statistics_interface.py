"""
Gradio interface for visualizing the policy of a model.
"""

import gradio as gr

from demo import utils, visualisation
from lczerolens import GameDataset
from lczerolens.xai import HasThreatConcept, UniqueConceptDataset

current_policy_statistics = None
current_lrp_statistics = None
current_probing_statistics = None
dataset = GameDataset("assets/test_stockfish_10.jsonl")
check_concept = HasThreatConcept("K", relative=True)
unique_check_dataset = UniqueConceptDataset.from_game_dataset(
    dataset, check_concept
)


def list_models():
    """
    List the models in the model directory.
    """
    models_info = utils.get_models_info(leela=False)
    return sorted([[model_info[0]] for model_info in models_info])


def on_select_model_df(
    evt: gr.SelectData,
):
    """
    When a model is selected, update the statement.
    """
    return evt.value


def compute_policy_statistics(
    model_name,
):
    global current_policy_statistics
    global dataset

    if model_name == "":
        gr.Warning(
            "Please select a model.",
        )
        return None
    wrapper, lens = utils.get_wrapper_lens_from_state(model_name, "policy")
    current_policy_statistics = lens.compute_statistics(dataset, wrapper, 10)
    return make_policy_plot()


def make_policy_plot():
    global current_policy_statistics

    if current_policy_statistics is None:
        gr.Warning(
            "Please compute policy statistics first.",
        )
        return None
    else:
        return visualisation.render_policy_statistics(
            current_policy_statistics
        )


def compute_lrp_statistics(
    model_name,
):
    global current_lrp_statistics
    global dataset

    if model_name == "":
        gr.Warning(
            "Please select a model.",
        )
        return None, None, None
    wrapper, lens = utils.get_wrapper_lens_from_state(model_name, "lrp")
    current_lrp_statistics = lens.compute_statistics(dataset, wrapper, 10)
    return make_lrp_plot()


def make_lrp_plot():
    global current_lrp_statistics

    if current_lrp_statistics is None:
        gr.Warning(
            "Please compute LRP statistics first.",
        )
        return None, None, None
    else:
        return visualisation.render_relevance_proportion(
            current_lrp_statistics
        )


def compute_probing_statistics(
    model_name,
):
    global current_probing_statistics
    global check_concept
    global unique_check_dataset

    if model_name == "":
        gr.Warning(
            "Please select a model.",
        )
        return None
    wrapper, lens = utils.get_wrapper_lens_from_state(
        model_name, "probing", concept=check_concept
    )
    current_probing_statistics = lens.compute_statistics(
        unique_check_dataset, wrapper, 10
    )
    return make_probing_plot()


def make_probing_plot():
    global current_probing_statistics

    if current_probing_statistics is None:
        gr.Warning(
            "Please compute probing statistics first.",
        )
        return None
    else:
        return visualisation.render_probing_statistics(
            current_probing_statistics
        )


with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column(scale=2):
            model_df = gr.Dataframe(
                headers=["Available models"],
                datatype=["str"],
                interactive=False,
                type="array",
                value=list_models,
            )
        with gr.Column(scale=1):
            with gr.Row():
                model_name = gr.Textbox(
                    label="Selected model", lines=1, interactive=False, scale=7
                )
    model_df.select(
        on_select_model_df,
        None,
        model_name,
    )

    with gr.Row():
        with gr.Column():
            policy_plot = gr.Plot(label="Policy statistics")
            policy_compute_button = gr.Button(
                value="Compute policy statistics"
            )
            policy_plot_button = gr.Button(value="Plot policy statistics")

            policy_compute_button.click(
                compute_policy_statistics,
                inputs=[model_name],
                outputs=[policy_plot],
            )
            policy_plot_button.click(make_policy_plot, outputs=[policy_plot])

        with gr.Column():
            lrp_plot_hist = gr.Plot(label="LRP history statistics")

    with gr.Row():
        with gr.Column():
            lrp_plot_planes = gr.Plot(label="LRP planes statistics")

        with gr.Column():
            lrp_plot_pieces = gr.Plot(label="LRP pieces statistics")

    with gr.Row():
        lrp_compute_button = gr.Button(value="Compute LRP statistics")
    with gr.Row():
        lrp_plot_button = gr.Button(value="Plot LRP statistics")

    lrp_compute_button.click(
        compute_lrp_statistics,
        inputs=[model_name],
        outputs=[lrp_plot_hist, lrp_plot_planes, lrp_plot_pieces],
    )
    lrp_plot_button.click(
        make_lrp_plot,
        outputs=[lrp_plot_hist, lrp_plot_planes, lrp_plot_pieces],
    )

    with gr.Column():
        probing_plot = gr.Plot(label="Probing statistics")
        probing_compute_button = gr.Button(value="Compute probing statistics")
        probing_plot_button = gr.Button(value="Plot probing statistics")

        probing_compute_button.click(
            compute_probing_statistics,
            inputs=[model_name],
            outputs=[probing_plot],
        )
        probing_plot_button.click(make_probing_plot, outputs=[probing_plot])
