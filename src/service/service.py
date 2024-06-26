from abc import ABC, abstractmethod
from typing import Callable, List, Dict

import torch
from lightning.pytorch.loggers import TensorBoardLogger


class Service(ABC):
    def __init__(self, config):
        self.config = config
        self.model_name = f"{self.config['APP']['ARCH']}"
        self.batch_size = int(self.config['APP']['BATCH_SIZE'])
        self.cpu_workers = int(self.config['APP']['CPU_WORKERS'])
        self.dataset_size_percent = float(self.config['APP']['DATA_SUBSET_SIZE_PERCENT'])

        self.noisy_mean = float(self.config['DATA']['NOISY_MEAN'])
        self.noisy_std = float(self.config['DATA']['NOISY_STD'])
        self.clean_mean = float(self.config['DATA']['CLEAN_MEAN'])
        self.clean_std = float(self.config['DATA']['CLEAN_STD'])

        self.memo = {}

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
