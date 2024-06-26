import lightning as pl

from src.module.adv_segan import AdvSEGAN
from src.module.segan_discriminator import SEGAN_Discriminator
from src.module.segan_residual_generator_bn import SEGAN_Residual_BN_Generator
from src.script.segan_script import SEGAN_Script


class Adv_SEGAN_Script(SEGAN_Script):

    def create_architecture(self, datamodule: pl.LightningDataModule):
        """
        Creates the AdvancedSEGAN architecture using the provided datamodule.

        This function retrieves a batch of reference input data from the datamodule,
        creates the AdvancedSEGAN generator and discriminator modules, and then returns the AdvancedSEGAN model.

        Args:
            datamodule (pl.LightningDataModule): The datamodule containing the data used for training.

        Returns:
            advanced_segan (AdvancedSEGAN): The AdvancedSEGAN model.
        """
        # Get a batch of reference input data
        # The first element of the tuple is the input data, which we'll use to create the generator and discriminator
        ref_batch_X, ref_batch_y = next(iter(datamodule.train_dataloader()))

        # Create the AdvancedSEGAN generator module
        generator = SEGAN_Residual_BN_Generator()

        # Create the AdvancedSEGAN discriminator module
        # The discriminator is initialized with the output of the generator on the reference data
        discriminator = SEGAN_Discriminator(ref_batch_X.shape[0], ref_batch_X.shape[-1])

        # Get model hyperparameters from the service object, or use an empty dictionary if not available
        model_hyperparams = self.service.model_hyperparams if hasattr(self.service, 'model_hyperparams') else {}

        # Create the AdvancedSEGAN model
        advanced_segan = AdvSEGAN(self.service, generator, discriminator,
                                  example_input_array=(ref_batch_X, ref_batch_y),
                                  noisy_mean=self.service.noisy_mean,
                                  noisy_std=self.service.noisy_std,
                                  clean_mean=self.service.clean_mean,
                                  clean_std=self.service.clean_std, **model_hyperparams)

        return advanced_segan
