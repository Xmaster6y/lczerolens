"""
Gradio interface for converting models.
"""

import os
import uuid

import gradio as gr

from demo import constants, utils
from lczerolens.utils import lczero as lczero_utils


def list_models():
    """
    List the models in the model directory.
    """
    models_info = utils.get_models_info()
    return sorted([[model_info[0]] for model_info in models_info])


def on_select_model_df(
    evt: gr.SelectData,
):
    """
    When a model is selected, update the statement.
    """
    return evt.value


def convert_model(
    model_name: str,
):
    """
    Convert the model.
    """
    if model_name == "":
        gr.Warning(
            "Please select a model.",
        )
        return list_models(), ""
    if model_name.endswith(".onnx"):
        gr.Warning(
            "ONNX conversion not implemented.",
        )
        return list_models(), ""
    try:
        lczero_utils.convertnet(
            f"{constants.LEELA_MODEL_DIRECTORY}/{model_name}",
            f"{constants.MODEL_DIRECTORY}/{model_name[:-6]}.onnx",
        )
    except RuntimeError:
        gr.Warning(
            f"Could not convert net at `{model_name}`.",
        )
        return list_models(), "Conversion failed"
    return list_models(), "Conversion successful"


def upload_model(
    model_file: gr.File,
):
    """
    Convert the model.
    """
    if model_file is None:
        gr.Warning(
            "File not uploaded.",
        )
        return list_models()
    try:
        id = uuid.uuid4()
        tmp_file_path = f"{constants.LEELA_MODEL_DIRECTORY}/{id}"
        with open(
            tmp_file_path,
            "wb",
        ) as f:
            f.write(model_file)
        utils.save_model(tmp_file_path)
    except RuntimeError:
        gr.Warning(
            "Invalid file type.",
        )
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
    return list_models()


def get_model_description(
    model_name: str,
):
    """
    Get the model description.
    """
    if model_name == "":
        gr.Warning(
            "Please select a model.",
        )
        return ""
    if model_name.endswith(".onnx"):
        gr.Warning(
            "ONNX description not implemented.",
        )
        return ""
    try:
        description = lczero_utils.describenet(
            f"{constants.LEELA_MODEL_DIRECTORY}/{model_name}",
        )
    except RuntimeError:
        raise gr.Error(
            f"Could not describe net at `{model_name}`.",
        )
    return description


def get_model_path(
    model_name: str,
):
    """
    Get the model path.
    """
    if model_name == "":
        gr.Warning(
            "Please select a model.",
        )
        return None
    if model_name.endswith(".onnx"):
        return f"{constants.MODEL_DIRECTORY}/{model_name}"
    else:
        return f"{constants.LEELA_MODEL_DIRECTORY}/{model_name}"


with gr.Blocks() as interface:
    model_file = gr.File(type="binary")
    upload_button = gr.Button(
        value="Upload",
    )
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
            conversion_status = gr.Textbox(
                label="Conversion status",
                lines=1,
                interactive=False,
            )

    convert_button = gr.Button(
        value="Convert",
    )
    describe_button = gr.Button(
        value="Describe model",
    )
    model_description = gr.Textbox(
        label="Model description",
        lines=1,
        interactive=False,
    )
    download_button = gr.Button(
        value="Get download link",
    )
    download_file = gr.File(
        type="filepath",
        label="Download link",
        interactive=False,
    )

    model_df.select(
        on_select_model_df,
        None,
        model_name,
    )
    upload_button.click(
        upload_model,
        model_file,
        model_df,
    )
    convert_button.click(
        convert_model,
        model_name,
        [model_df, conversion_status],
    )
    describe_button.click(
        get_model_description,
        model_name,
        model_description,
    )
    download_button.click(
        get_model_path,
        model_name,
        download_file,
    )
