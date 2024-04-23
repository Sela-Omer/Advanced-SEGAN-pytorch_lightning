from abc import ABC, abstractmethod
from typing import Callable, List, Dict

import torch
from lightning.pytorch.loggers import TensorBoardLogger


class Service(ABC):
    def __init__(self, config):
        self.config = config
        self.model_name = f"{self.config['APP']['ARCH']}-{self.config['APP']['VERSION']}v"
        self.batch_size = int(self.config['APP']['BATCH_SIZE'])
        self.dataset_size_percent = float(self.config['APP']['DATA_SUBSET_SIZE_PERCENT'])

    @property
    @abstractmethod
    def scripts(self) -> Dict[str, Callable]:
        """
        Returns a dictionary of scripts.
        The dictionary contains a single key-value pair.
        The key is the name of the script, and the value is a function that implements the script.
        Returns:
            Dict[str, Callable]: A dictionary of scripts.
        """
        pass
