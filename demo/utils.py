"""
Utils for the demo app.
"""

import os
import re
import subprocess

from demo import constants, state
from lczerolens.game import LczerroModelWrapper
from lczerolens.utils import lczero as lczero_utils
from lczerolens.xai import AttentionWrapper


def get_models_info(leela=True):
    """
    Get the names of the models in the model directory.
    """
    model_df = []
    exp = r"(?P<n_filters>\d+)x(?P<n_blocks>\d+)"
    for filename in os.listdir(constants.MODEL_DIRECTORY):
        if filename.endswith(".onnx"):
            match = re.search(exp, filename)
            if match is None:
                n_filters = -1
                n_blocks = -1
            else:
                n_filters = int(match.group("n_filters"))
                n_blocks = int(match.group("n_blocks"))
            model_df.append(
                [
                    filename,
                    "ONNX",
                    n_blocks,
                    n_filters,
                ]
            )
    if leela:
        for filename in os.listdir(constants.LEELA_MODEL_DIRECTORY):
            if filename.endswith(".pb.gz"):
                match = re.search(exp, filename)
                if match is None:
                    n_filters = -1
                    n_blocks = -1
                else:
                    n_filters = int(match.group("n_filters"))
                    n_blocks = int(match.group("n_blocks"))
                model_df.append(
                    [
                        filename,
                        "LEELA",
                        n_blocks,
                        n_filters,
                    ]
                )
    return model_df


def save_model(tmp_file_path):
    """
    Save the model to the model directory.
    """
    popen = subprocess.Popen(
        ["file", tmp_file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    popen.wait()
    if popen.returncode != 0:
        raise RuntimeError
    file_desc = (
        popen.stdout.read().decode("utf-8").split(tmp_file_path)[1].strip()
    )
    rename_match = re.search(r"was\s\"(?P<name>.+)\"", file_desc)
    type_match = re.search(r"\:\s(?P<type>[a-zA-Z]+)", file_desc)
    if rename_match is None or type_match is None:
        raise RuntimeError
    model_name = rename_match.group("name")
    model_type = type_match.group("type")
    if model_type != "gzip":
        raise RuntimeError
    os.rename(
        tmp_file_path,
        f"{constants.LEELA_MODEL_DIRECTORY}/{model_name}.gz",
    )
    try:
        lczero_utils.describenet(
            f"{constants.LEELA_MODEL_DIRECTORY}/{model_name}.gz",
        )
    except RuntimeError:
        os.remove(f"{constants.LEELA_MODEL_DIRECTORY}/{model_name}.gz")
        raise RuntimeError


def get_wrapper_from_state(model_name):
    """
    Get the model wrapper from the state.
    """
    if model_name in state.models:
        wrapper = LczerroModelWrapper.from_model(state.models[model_name])
        return wrapper
    else:
        wrapper = LczerroModelWrapper(
            f"{constants.MODEL_DIRECTORY}/{model_name}"
        )
        state.models[model_name] = wrapper.model
        return wrapper


def get_attention_wrapper_from_state(model_name):
    """
    Get the model wrapper from the state.
    """
    if model_name in state.models:
        wrapper = AttentionWrapper.from_model(state.models[model_name])
        return wrapper
    else:
        wrapper = AttentionWrapper(f"{constants.MODEL_DIRECTORY}/{model_name}")
        state.models[model_name] = wrapper.model
        return wrapper
