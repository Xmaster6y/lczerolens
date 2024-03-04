"""
Global state for the demo application.
"""

from typing import Dict

from lczerolens import Lens, ModelWrapper

wrappers: Dict[str, ModelWrapper] = {}

lenses: Dict[str, Dict[str, Lens]] = {
    "activation": {},
    "lrp": {},
    "crp": {},
    "policy": {},
    "probing": {},
    "patching": {},
}
