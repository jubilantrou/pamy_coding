"""The collection of all useful funcitons
"""
import numpy as np
import os
from typing import Dict
import pathlib


def mkdir(path: pathlib.Path) -> None:
    """Create the file if it does not exist
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

