"""
Global state for the demo application.
"""

from typing import Dict

import torch

models: Dict[str, torch.nn.Module] = {}
