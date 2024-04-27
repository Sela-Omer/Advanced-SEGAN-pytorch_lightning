from abc import ABC, abstractmethod
from typing import Callable

import lightning as pl

from src.service.service import Service


class Script(ABC, Callable):
    """
    Class for setting up the script for a given service.

    Attributes:
        service (Service): An instance of the Service class.
    """

    def __init__(self, service: Service):
        """
        Initializes the Script class with a given service.

        Parameters:
            service (Service): An instance of the Service class.
        """
        self.service = service

    @abstractmethod
    def create_datamodule(self):
        """
        Create the data module for the training script.

        This method is an abstract method that should be implemented by subclasses. It is responsible for creating and returning
        the data module that will be used for training. The data module should be an instance of a class that inherits from
        `pl.LightningDataModule`.

        Returns:
            pl.LightningDataModule: The data module for the training script.
        """
        pass

    @abstractmethod
    def create_architecture(self, datamodule: pl.LightningDataModule):
        """
        Creates the architecture for the model.

        This is an abstract method that should be implemented by subclasses. It is responsible for creating and returning the architecture of the model. The architecture is defined by the specific subclass and should be based on the provided `datamodule` which contains the data used for training.

        Parameters:
            datamodule (pl.LightningDataModule): The data module containing the data used for training.

        Returns:
            pl.LightningModule: The architecture of the model.
        """
        pass

    @abstractmethod
    def create_trainer(self, callbacks: list):
        """
        Create a trainer with specified configurations.

        Args:
            callbacks (list): A list of callbacks to be used during training.

        Returns:
            pl.Trainer: The created trainer object.
        """
        pass

    @abstractmethod
    def __call__(self):
        pass
