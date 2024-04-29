from abc import ABC
from typing import Tuple

import lightning as pl
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data import DataLoader

from src.datamodule.audio_data_module import AudioDataModule
from src.dataset.map_audio_dataset import MapAudioDataset
from src.script.script import Script


class StatsScript(Script, ABC):

    def create_trainer(self, callbacks: list):
        pass

    def create_architecture(self, datamodule: pl.LightningDataModule):
        pass

    def create_datamodule(self):
        """
        Create and return an instance of the AudioDataModule.

        This method creates an instance of the AudioDataModule using the provided service, data directory, and audio dataset class.
        If the application environment is set to 'DEVELOPMENT', it also prints the representation of the train dataset.

        Returns:
            AudioDataModule: The created instance of the AudioDataModule.
        """
        # Get the data directory and audio dataset class from the service configuration
        data_dir = self.service.config['DATA']['PATH']
        audio_dataset_class = MapAudioDataset

        # Create an instance of the AudioDataModule
        datamodule = AudioDataModule(service=self.service, data_dir=data_dir, audio_dataset_class=audio_dataset_class)

        # If the application environment is set to 'DEVELOPMENT', print the representation of the train dataset
        if self.service.config['APP']['ENVIRONMENT'] == 'DEVELOPMENT':
            datamodule.prepare_data()
            datamodule.setup(stage='fit')
            datamodule.train_dataset.__repr__()

        return datamodule

    def generate_stats_dataloader(self, audio_dataloader: DataLoader) -> Tuple[float, float, float, float]:
        """
        Generate statistics for the audio data in the dataloader.

        This function iterates over the audio_dataloader, generating statistics for each batch of audio data.
        The statistics consist of the mean and standard deviation of the audio data.

        Args:
            audio_dataloader (torch.utils.data.DataLoader): The dataloader containing audio data.

        Returns:
            Tuple[float, float, float, float]: A tuple containing the mean and standard deviation of the audio data for X and y respectively.
        """

        # Initialize lists to store the statistics for X and y
        X_stats, y_stats = [], []

        # Iterate over the dataloader
        for X, y in audio_dataloader:
            # Generate statistics for X and y
            # generate_stats_one_batch function is assumed to return a tuple (mean, std)
            X_stats.append(self.generate_stats_one_batch(X))
            y_stats.append(self.generate_stats_one_batch(y))

        # Convert the lists to numpy arrays and select the first two columns
        # Assuming generate_stats_one_batch returns (mean, std, num_samples)
        # We select only the mean and std for each batch
        X_stats = np.array(X_stats)[:, :2]
        y_stats = np.array(y_stats)[:, :2]

        # Calculate the mean of the statistics for X and y
        X_mean, X_std = X_stats.mean(axis=0)
        y_mean, y_std = y_stats.mean(axis=0)

        # Return the mean and standard deviation of the audio data for X and y respectively
        return X_mean, X_std, y_mean, y_std

    def generate_stats_one_batch(self, audio_batch: Tensor):
        """
        Generate statistics for a single batch of audio data.

        Args:
            audio_batch (torch.Tensor): A tensor of shape (batch_size, channels, samples) containing audio data.

        Returns:
            Tuple[float, float, int]: A tuple containing the mean, standard deviation, and number of samples in the audio batch.
        """
        # Ensure that the audio batch has the expected shape
        assert audio_batch.dim() == 3, "Audio batch must have shape (batch_size, channels, samples)"
        assert audio_batch.size(1) == 1, "Audio batch must have one channel"

        # Calculate the number of samples in the batch
        num_samples = audio_batch.size(0) * audio_batch.size(2)

        # Calculate the mean and standard deviation of the audio batch
        mean, std = audio_batch.mean().item(), audio_batch.std().item()

        # Return the mean, standard deviation, and number of samples in the audio batch
        return mean, std, num_samples

    def __call__(self):
        """
        This method orchestrates the training process.
        It creates the data module, architecture, callbacks, and trainer,
        and then fits the model using the trainer.
        """
        # Create the data module
        datamodule = self.create_datamodule()

        train_dl = datamodule.train_dataloader()

        # Generate statistics for the audio data in the dataloader
        X_mean, X_std, y_mean, y_std = self.generate_stats_dataloader(train_dl)

        # Print the statistics
        print(f"X_mean={X_mean},X_std={X_std},y_mean={y_mean},y_std={y_std}")
