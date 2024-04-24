import lightning as pl
import torch
import torch.optim as optim
from torch import nn

from src.service.service import Service


class SEGAN(pl.LightningModule):
    """
    This class represents the SEGAN model. It inherits from the PyTorch Lightning Module class.

    Args:
        generator (nn.Module): The generator neural network module.
        discriminator (nn.Module): The discriminator neural network module.
        lr_gen (float): The learning rate for the generator.
        lr_disc (float): The learning rate for the discriminator.
        lambda_l1 (int): The lambda value for the L1 loss.

    Example:
        >>> from src.module.segan_discriminator import SEGAN_Discriminator
        >>> from src.module.segan_generator import SEGAN_Generator
        >>> generator = SEGAN_Generator(encoder_dimensions=[1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024])
        >>> discriminator = SEGAN_Discriminator(torch.rand((400,2,16384)), n_out_features=1, discriminator_dimensions=[2, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024], vbn_epsilon=1e-5)
        >>> model = SEGAN(generator, discriminator, lr_gen=0.0002, lr_disc=0.0002, lambda_l1=100)

    """

    def __init__(self, service: Service, generator: nn.Module, discriminator: nn.Module, lr_gen=0.0002, lr_disc=0.0002,
                 lambda_l1=100, example_input_array=None):
        """
        Initializes the SEGAN model with the provided generator, discriminator, learning rates, and lambda value for L1 loss.

        Args:
            generator (nn.Module): The generator neural network module.
            discriminator (nn.Module): The discriminator neural network module.
            lr_gen (float): The learning rate for the generator.
            lr_disc (float): The learning rate for the discriminator.
            lambda_l1 (int): The lambda value for the L1 loss.

        """
        super().__init__()
        self.example_input_array = example_input_array
        self.service = service

        self.automatic_optimization = False

        # Assigning the provided parameters to the instance variables
        self.generator = generator
        self.discriminator = discriminator
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.lambda_l1 = lambda_l1

        # Initializing the loss functions
        self.criterionGAN = nn.MSELoss()  # Using Mean Squared Error Loss for LSGAN
        self.criterionL1 = nn.L1Loss()

    def forward(self, noisy, clean):
        """
        Apply the generator neural network to the noisy input.

        Args:
            noisy (torch.Tensor): The noisy input tensor.

        Returns:
            torch.Tensor: The generated cleaned version of the noisy input.
        """
        # Apply the generator neural network to the noisy input
        generated_clean = self.generator(noisy)

        return generated_clean

    def training_step(self, batch):
        """
        Perform a training step.
        Args:
            batch (tuple): A tuple containing the noisy and clean audio samples.
        """
        g_opt, d_opt = self.optimizers()

        # Unpack the batch
        noisy, clean = batch

        # Generate a cleaned version of the noisy audio
        generated_clean = self.generator(noisy)

        # Define targets for real and fake examples
        valid = torch.ones(noisy.size(0), 1, device=self.device)
        fake = torch.zeros(noisy.size(0), 1, device=self.device)

        ######################
        # Optimize Generator #
        ######################
        # The generator's goal is to fool the discriminator
        fake_pred = self.discriminator(generated_clean)
        g_loss = 0.5 * self.criterionGAN(fake_pred, valid)

        # Generator tries to get discriminator to output valid
        l1_loss = self.criterionL1(generated_clean[:, 0].unsqueeze(1), clean)

        g_total_loss = g_loss + self.lambda_l1 * l1_loss
        self.log('g_loss', g_loss)
        self.log('g_l1_loss', l1_loss)
        self.log('g_total_loss', g_total_loss)

        g_opt.zero_grad()
        self.manual_backward(g_total_loss)
        g_opt.step()

        ##########################
        # Optimize Discriminator #
        ##########################
        # How well can the discriminator differentiate clean sounds?
        real_pred = self.discriminator(torch.cat((clean, noisy), dim=1))
        fake_pred = self.discriminator(generated_clean.detach())

        # Discriminator tries to recognize real as valid
        real_loss = self.criterionGAN(real_pred, valid)

        # Discriminator tries to recognize fake as fake
        fake_loss = self.criterionGAN(fake_pred, fake)

        d_loss = (real_loss + fake_loss) / 2
        self.log('d_real_loss', real_loss)
        self.log('d_fake_loss', fake_loss)
        self.log('d_loss', d_loss)

        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.
        Args:
            batch (tuple): A tuple containing the noisy and clean audio samples.
        """
        self.log(self.service.config['FIT']['CHECKPOINT_MONITOR'], 0)
        return 0

    def configure_optimizers(self):
        """
        This method configures the optimizers for the generator and discriminator.

        Returns:
            list: A list containing the optimizers for the generator and discriminator.
        """
        # Create an RMSprop optimizer for the generator
        optimizer_gen = optim.RMSprop(
            self.generator.parameters(),
            lr=self.lr_gen
        )

        # Create an RMSprop optimizer for the discriminator
        optimizer_disc = optim.RMSprop(
            self.discriminator.parameters(),
            lr=self.lr_disc
        )

        return [optimizer_gen, optimizer_disc]
