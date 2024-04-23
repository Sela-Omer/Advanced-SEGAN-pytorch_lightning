from typing import Callable, Dict

import torch
from lightning.pytorch.loggers import TensorBoardLogger

from src.script.segan_fit_script import SEGAN_FitScript
from src.service.service import Service


class ServiceFit(Service):
    """
    Class for setting up the training script for a given service.

    Attributes:
        service (ServiceFit): An instance of the ServiceFit class.
    """

    def __init__(self, config):
        super().__init__(config)

        torch.set_float32_matmul_precision(config['FIT-PARAMS']['TORCH_PRECISION'])

    @property
    def scripts(self) -> Dict[str, Callable]:
        """
        Returns a dictionary of scripts.

        The dictionary contains a single key-value pair.
        The key is the name of the ARCH , and the value is an instance of the Fit Script class.
        """
        # Create a dictionary with the ARCH name and its corresponding instance
        script_dict = {
            'SEGAN': SEGAN_FitScript(self)  # Instantiate the SEGAN_Fit class
        }

        return script_dict
