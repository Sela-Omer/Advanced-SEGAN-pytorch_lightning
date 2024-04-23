from abc import ABC, abstractmethod
from typing import Callable

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from src.service.service import Service


class FitScript(ABC, Callable):
    """
    Class for setting up the training script for a given service.

    Attributes:
        service (ServiceFit): An instance of the ServiceFit class.
    """

    def __init__(self, service: Service):
        """
        Initializes the Fit_Script class with a given service.

        Parameters:
            service (ServiceFit): An instance of the ServiceFit class.
        """
        # Assign the provided service to the instance variable
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

    def create_callbacks(self):
        """
        Creates a list of callbacks for the model training process.

        Returns:
            list: A list of callbacks, including a model checkpoint callback, a progress bar callback, and a device stats callback.
        """
        # Create the model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor=self.service.config['FIT']['CHECKPOINT_MONITOR'],  # Metric to monitor
            filename=self.service.model_name + '-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,  # Save the top 3 models
            mode='min',  # Minimize the monitored metric (val_loss)
        )

        # Create the progress bar callback
        progress_bar_callback = TQDMProgressBar()

        # Create the device stats callback
        device_stats_callback = DeviceStatsMonitor()

        # Create the list of callbacks
        return [checkpoint_callback, progress_bar_callback, device_stats_callback]

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

    def create_trainer(self, callbacks: list):
        """
        Create a trainer with specified configurations.

        Args:
            callbacks (list): A list of callbacks to be used during training.

        Returns:
            pl.Trainer: The created trainer object.
        """
        # Create the trainer with specified configurations
        trainer = pl.Trainer(
            max_epochs=int(self.service.config['FIT']['N_EPOCHS']),
            accelerator=self.service.config['FIT']['ACCELERATOR'],
            log_every_n_steps=int(self.service.config['FIT']['LOG_EVERY_N_STEPS']),
            callbacks=callbacks,
            logger=TensorBoardLogger(save_dir=self.service.config['FIT']['MODEL_STORE_PATH'],
                                     name=self.service.model_name),
            gpus=int(self.service.config['FIT']['GPUS']),
            num_nodes=int(self.service.config['FIT']['NUM_NODES']),
        )
        return trainer

    def __call__(self):
        """
        This method orchestrates the training process.
        It creates the data module, architecture, callbacks, and trainer,
        and then fits the model using the trainer.
        """
        # Create the data module
        datamodule = self.create_datamodule()

        # Create the architecture
        arch = self.create_architecture(datamodule)

        # Create the callbacks
        callbacks = self.create_callbacks()

        # Create the trainer
        trainer = self.create_trainer(callbacks)

        # Fit the model using the trainer
        trainer.fit(arch, datamodule=datamodule)
