from abc import ABC

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from src.script.script import Script


class FitScript(Script, ABC):
    """
    Class for setting up the training script for a given service.

    Attributes:
        service (ServiceFit): An instance of the ServiceFit class.
    """

    def create_callbacks(self):
        """
        Creates a list of callbacks for the model training process.

        Returns:
            list: A list of callbacks, including a model checkpoint callback, a progress bar callback, and a device stats callback.
        """

        checkpoint_callback_lst = []
        for metric_name in self.service.config['FIT']['CHECKPOINT_MONITOR'].split(','):
            # Create the model checkpoint callback
            checkpoint_callback_lst.append(ModelCheckpoint(
                monitor=metric_name,  # Metric to monitor
                filename=self.service.model_name + '-{epoch:02d}-{' + metric_name + ':.2f}',
                save_top_k=3,  # Save the top 3 checkpoints
                mode='max',  # Maximize the monitored metric
            ))

        # Create the progress bar callback
        progress_bar_callback = TQDMProgressBar()

        # Create the device stats callback
        device_stats_callback = DeviceStatsMonitor()

        # Create the list of callbacks
        return checkpoint_callback_lst + [progress_bar_callback, device_stats_callback]

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
            accelerator=self.service.config['APP']['ACCELERATOR'],
            log_every_n_steps=int(self.service.config['FIT']['LOG_EVERY_N_STEPS']),
            callbacks=callbacks,
            logger=TensorBoardLogger(save_dir=self.service.config['APP']['MODEL_STORE_PATH'],
                                     name=self.service.model_name,
                                     log_graph=bool(self.service.config['FIT']['LOG_GRAPH'])),
            devices=int(self.service.config['APP']['DEVICES']),
            num_nodes=int(self.service.config['APP']['NUM_NODES']),
            strategy=self.service.config['APP']['STRATEGY'],
            precision=self.service.config['FIT']['TRAINER_PRECISION'],
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
