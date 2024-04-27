import lightning as pl

from src.datamodule.audio_data_module import AudioDataModule
from src.dataset.map_audio_dataset import MapAudioDataset
from src.module.segan import SEGAN
from src.module.segan_discriminator import SEGAN_Discriminator
from src.module.segan_generator import SEGAN_Generator
from src.script.script import Script


class SEGAN_Script(Script):
    """
    This module contains the code for training a SEGAN model using the PyTorch Lightning framework.

    The SEGAN (Sound Exaggeration Generative Adversarial Network) is a generative model that can generate audio samples.

    The training process involves creating the data module, loading the data, and defining the architecture of the SEGAN model.

    The data module, `AudioDataModule`, is responsible for loading and preprocessing the data. It also contains the logic for downloading and extracting the dataset.

    """

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

    def create_architecture(self, datamodule: pl.LightningDataModule):
        """
        Creates the SEGAN architecture using the provided datamodule.

        This function retrieves a batch of reference input data from the datamodule,
        creates the SEGAN generator and discriminator modules, and then returns the SEGAN model.

        Args:
            datamodule (pl.LightningDataModule): The datamodule containing the data used for training.

        Returns:
            segan (SEGAN): The SEGAN model.
        """
        # Get a batch of reference input data
        # The first element of the tuple is the input data, which we'll use to create the generator and discriminator
        ref_batch_X, ref_batch_y = next(iter(datamodule.train_dataloader()))

        # Create the SEGAN generator module
        generator = SEGAN_Generator()

        # Create the SEGAN discriminator module
        # The discriminator is initialized with the output of the generator on the reference data
        discriminator = SEGAN_Discriminator(ref_batch_X.shape[0], ref_batch_X.shape[-1])

        model_hyperparams = self.service.model_hyperparams if hasattr(self.service, 'model_hyperparams') else {}

        # Create the SEGAN model
        segan = SEGAN(self.service, generator, discriminator, example_input_array=(ref_batch_X, ref_batch_y),
                      **model_hyperparams)

        return segan
