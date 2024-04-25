from typing import Callable, Dict

import torch

from src.helper.param_helper import convert_param_to_type
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

        torch.set_float32_matmul_precision(config['FIT']['TORCH_PRECISION'])

        self.model_hyperparams = {}
        hyperparam_lst = config['FIT']['MODEL_HYPERPARAMS'].split(',')
        for hyperparam in hyperparam_lst:
            if '=' not in hyperparam:
                continue
            key, value = hyperparam.split('=')
            self.model_hyperparams[key] = convert_param_to_type(value)

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
