from abc import ABC

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from src.script.script import Script


class EvalScript(Script, ABC):

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
            max_epochs=1,
            accelerator=self.service.config['APP']['ACCELERATOR'],
            log_every_n_steps=1,
            callbacks=callbacks,
            logger=TensorBoardLogger(save_dir=self.service.config['APP']['MODEL_STORE_PATH'],
                                     name=self.service.model_name,
                                     log_graph=True),
            devices=int(self.service.config['APP']['DEVICES']),
            num_nodes=int(self.service.config['APP']['NUM_NODES']),
            strategy=self.service.config['APP']['STRATEGY'],
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

        # Create the trainer
        trainer = self.create_trainer([])

        # Fit the model using the trainer
        trainer.test(model=arch, datamodule=datamodule, ckpt_path=self.service.config['EVAL']['CHECKPOINT_PATH'],
                     verbose=True)
