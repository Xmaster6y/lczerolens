"""
Global state for the demo application.
"""

from typing import Dict

from lczerolens.adapt import ModelWrapper
from lczerolens.xai import AttentionLens, LrpLens

wrappers: Dict[str, ModelWrapper] = {}
attention_lenses: Dict[str, AttentionLens] = {}
lrp_lenses: Dict[str, LrpLens] = {}
