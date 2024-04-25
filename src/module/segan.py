import lightning as pl
import numpy as np
import torch
import torch.optim as optim
from torch import nn

from src.helper.metrics_helper import composite_metrics
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
        >>> model = SEGAN(generator, discriminator, lr_gen=0.0001, lr_disc=0.0001, lambda_l1=100)

    """

    def __init__(self, service: Service, generator: nn.Module, discriminator: nn.Module, lr_gen=0.0002, lr_disc=0.0002,
                 lambda_l1=100, example_input_array=None, sample_rate=16000):
        """
        Initializes the SEGAN model with the provided generator, discriminator, learning rates, and lambda value for L1 loss.

        Args:
            generator (nn.Module): The generator neural network module.
            discriminator (nn.Module): The discriminator neural network module.
            lr_gen (float): The learning rate for the generator.
            lr_disc (float): The learning rate for the discriminator.
            lambda_l1 (int): The lambda value for the L1 loss.

        """
        super(SEGAN, self).__init__()
        self.sample_rate = sample_rate
        self.example_input_array = example_input_array

        ref_batch = torch.cat(example_input_array, dim=1)
        self.register_buffer('ref_batch', ref_batch, persistent=True)

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

    def forward(self, noisy, clean, z=None):
        """
        Forward pass of the model.

        Args:
            noisy (torch.Tensor): The input tensor representing the noisy data.
            clean (torch.Tensor): The input tensor representing the clean data.
            z (torch.Tensor, optional): The thought vector tensor. Defaults to None.

        Returns:
            torch.Tensor: The generated clean data tensor.
        """
        if z is None:
            # create thought vector
            z = torch.zeros([noisy.size(0), 1024, 8]).to(device=noisy.device)

        # Apply the generator neural network to the noisy input
        generated_clean = self.generator(noisy, z.clone())

        return generated_clean

    def _get_targets_for_real_and_fake_examples(self, bs: int):
        """
        Generate targets for real and fake examples.

        Args:
            bs (int): The batch size.

        Returns:
            torch.Tensor: The valid targets for real examples.
            torch.Tensor: The fake targets for fake examples.
        """
        # Define targets for real and fake examples
        valid = torch.ones(bs, 1, device=self.device)
        fake = torch.zeros(bs, 1, device=self.device)

        return valid, fake

    def _compute_intermediate_step(self, batch, log_prefix='train'):
        """
        Compute the intermediate step for the given batch.

        Args:
            batch (tuple): A tuple containing the noisy and clean audio samples.
            log_prefix (str, optional): The prefix to be added to the log messages. Defaults to 'train'.

        Returns:
            dict: A dictionary containing the following keys:
                - generator_fake_pred (torch.Tensor): The predicted values of the generator for fake examples.
                - generator_raw_pred (torch.Tensor): The raw predicted values of the generator.
                - discriminator_real_pred (torch.Tensor): The predicted values of the discriminator for real examples.
                - discriminator_fake_pred (torch.Tensor): The predicted values of the discriminator for fake examples.
                - clean (torch.Tensor): The clean audio samples.
                - noisy (torch.Tensor): The noisy audio samples.
                - batch_size (int): The size of the batch.
                - ref_batch (torch.Tensor): The reference batch used for generating the clean audio.
                - generated_ref_batch (torch.Tensor): The generated reference batch.
                - generated_clean (torch.Tensor): The generated clean audio.
        """

        # Unpack the batch
        noisy, clean = batch

        # Create a detached copy of the reference batch
        ref_batch = self.ref_batch.clone().detach()

        # create thought vector
        ref_z = self._make_z([ref_batch.size(0), 1024, 8], device=noisy.device)
        z = ref_z[:noisy.size(0)].clone()

        # Apply generator to reference batch
        generated_ref_batch = self.generator(ref_batch[:, 0].unsqueeze(1), ref_z.clone()).detach()

        # Generate a cleaned version of the noisy audio
        if log_prefix is None:
            generated_clean = self.generator(noisy, z.clone())
        else:
            generated_clean = self.generator(noisy, z.clone(), log=self.log, log_prefix=f'{log_prefix}_generator')

        ##########################
        # Compute Generator Step #
        ##########################
        # The generator's goal is to fool the discriminator
        generator_fake_pred = self.discriminator(generated_clean, generated_ref_batch)
        generator_raw_pred = generated_clean[:, 0].unsqueeze(1)

        ##############################
        # Compute Discriminator Step #
        ##############################
        # How well can the discriminator differentiate clean sounds?
        discriminator_real_pred = self.discriminator(torch.cat((clean, noisy), dim=1), ref_batch)
        discriminator_fake_pred = self.discriminator(generated_clean.detach(), generated_ref_batch)

        return {
            'generator_fake_pred': generator_fake_pred,
            'generator_raw_pred': generator_raw_pred,
            'discriminator_real_pred': discriminator_real_pred,
            'discriminator_fake_pred': discriminator_fake_pred,
            'clean': clean,
            'noisy': noisy,
            'batch_size': noisy.size(0),
            'ref_batch': ref_batch,
            'generated_ref_batch': generated_ref_batch,
            'generated_clean': generated_clean
        }

    def _calc_generator_loss(self, intermediate_step: dict, valid: torch.Tensor) -> dict:
        """
        Calculate the generator loss for the SEGAN model.

        Args:
            intermediate_step (dict): A dictionary containing the intermediate step outputs.
            valid (torch.Tensor): A tensor containing the valid labels.

        Returns:
            dict: A dictionary containing the generator loss, L1 loss, and total loss.
        """
        # Unpack the intermediate step
        generator_fake_pred = intermediate_step['generator_fake_pred']
        generator_raw_pred = intermediate_step['generator_raw_pred']
        clean = intermediate_step['clean']

        # Calculate the generator loss
        g_loss = 0.5 * self.criterionGAN(generator_fake_pred, valid)

        # Calculate the L1 loss
        l1_loss = self.criterionL1(generator_raw_pred, clean)

        # Calculate the total loss
        g_total_loss = g_loss + self.lambda_l1 * l1_loss

        return {
            'g_loss': g_loss,
            'g_l1_loss': l1_loss,
            'g_total_loss': g_total_loss
        }

    def _calc_generator_composite_metrics(self, intermediate_step: dict) -> dict | None:
        """
        Calculate the composite metrics for the generator output.

        Args:
            intermediate_step (dict): A dictionary containing the intermediate step outputs.

        Returns:
            dict: A dictionary containing the composite metrics:
            - CSIG (float): Composite signal-to-interference+gradient ratio.
            - CBAK (float): Composite background quality.
            - COVL (float): Composite over-all quality.
            - wss_dist (float): Weighted signal to noise+distortion ratio.
            - llr_mean (float): Mean log-likelihood ratio.
            - PESQ (float): Perceptual Evaluation of Speech Quality.
            - SSNR (float): Signal-to-noise ratio segment-wise.
        """
        # Extract the generator raw prediction and clean audio from the intermediate step
        generator_raw_pred = intermediate_step['generator_raw_pred']
        clean = intermediate_step['clean']

        # Convert the clean and generator raw prediction tensors to numpy arrays
        clean_batch_arr = clean.detach().cpu().squeeze(1).numpy()
        generator_raw_pred_batch_arr = generator_raw_pred.detach().cpu().squeeze(1).numpy()

        comp_metrics = None
        count = 0
        for i in range(clean_batch_arr.shape[0]):
            clean_arr = clean_batch_arr[i]
            generator_raw_pred_arr = generator_raw_pred_batch_arr[i]
            try:
                comp_metrics_i = composite_metrics(clean_arr,
                                                   generator_raw_pred_arr,
                                                   self.sample_rate)

                if comp_metrics is None:
                    comp_metrics = comp_metrics_i
                else:
                    for key in comp_metrics:
                        comp_metrics[key] += comp_metrics_i[key]
                count += 1
            except:
                pass

        # If no utterances were processed, return None
        if comp_metrics is None:
            return None

        # Average the metrics across the batch
        for key in comp_metrics:
            comp_metrics[key] /= count

        return comp_metrics

    def _calc_discriminator_loss(self, intermediate_step: dict, valid: torch.Tensor, fake: torch.Tensor) -> dict:
        """
        Calculate the discriminator loss for the SEGAN model.

        Args:
            intermediate_step (dict): A dictionary containing the intermediate step outputs.
            valid (torch.Tensor): A tensor containing the valid labels.
            fake (torch.Tensor): A tensor containing the fake labels.

        Returns:
            dict: A dictionary containing the discriminator loss for real and fake examples,
                and the total discriminator loss.
        """
        # Unpack the intermediate step
        generated_clean = intermediate_step['generated_clean']
        generated_ref_batch = intermediate_step['generated_ref_batch']
        clean = intermediate_step['clean']
        noisy = intermediate_step['noisy']
        ref_batch = intermediate_step['ref_batch']

        # Calculate the discriminator predictions for real and fake examples
        real_pred = self.discriminator(torch.cat((clean, noisy), dim=1), ref_batch)
        fake_pred = self.discriminator(generated_clean.detach(), generated_ref_batch)

        # Discriminator tries to recognize real as valid
        d_real_loss = self.criterionGAN(real_pred, valid)

        # Discriminator tries to recognize fake as fake
        d_fake_loss = self.criterionGAN(fake_pred, fake)

        # Calculate the total discriminator loss
        d_total_loss = (d_real_loss + d_fake_loss) / 2

        return {
            'd_real_loss': d_real_loss,
            'd_fake_loss': d_fake_loss,
            'd_total_loss': d_total_loss
        }

    def _make_z(self, shape, mean=0.0, std=1.0, device=None):
        """
        Generate a random tensor with a normal distribution.

        Args:
            shape (Tuple[int]): The shape of the tensor.
            mean (float, optional): The mean of the normal distribution. Defaults to 0.0.
            std (float, optional): The standard deviation of the normal distribution. Defaults to 1.0.
            device (torch.device, optional): The device on which to place the tensor. Defaults to None, in which case the tensor is placed on the CPU.

        Returns:
            torch.Tensor: A tensor with the specified shape, mean, and standard deviation.
        """

        z = nn.init.normal_(torch.zeros(shape), mean=mean, std=std)

        return z.to(device if device else torch.device("cpu"))

    def training_step(self, batch):
        """
        Perform a training step.

        Args:
            batch (tuple): A tuple containing the noisy and clean audio samples.

        This function performs a training step for the SEGAN model. It first computes the intermediate step by calling the `_compute_intermediate_step` method. Then, it gets the targets for real and fake examples by calling the `_get_targets_for_real_and_fake_examples` method.

        After that, it optimizes the generator and discriminator. The generator loss is calculated by calling the `_calc_generator_loss` method, and the discriminator loss is calculated by calling the `_calc_discriminator_loss` method.

        The generator loss is logged using the `log` method, and the optimizer is updated and stepped. Similarly, the discriminator loss is logged and the optimizer is updated and stepped.
        """
        # Get the optimizers
        g_opt, d_opt = self.optimizers()

        # Compute the intermediate step
        intermediate_step = self._compute_intermediate_step(batch, 'train')

        # Get the targets for real and fake examples
        valid, fake = self._get_targets_for_real_and_fake_examples(intermediate_step['batch_size'])

        ######################
        # Optimize Generator #
        ######################

        # Calculate the generator loss
        generator_loss_dict = self._calc_generator_loss(intermediate_step, valid)

        # Log the generator loss
        self.log('train_g_loss', generator_loss_dict['g_loss'])
        self.log('train_g_l1_loss', generator_loss_dict['g_l1_loss'])
        self.log('train_g_total_loss', generator_loss_dict['g_total_loss'], prog_bar=True)

        # Zero the gradients and perform backpropagation
        g_opt.zero_grad()
        self.manual_backward(generator_loss_dict['g_total_loss'])
        g_opt.step()

        ##########################
        # Optimize Discriminator #
        ##########################

        # Calculate the discriminator loss
        discriminator_loss_dict = self._calc_discriminator_loss(intermediate_step, valid, fake)

        # Log the discriminator loss
        self.log('train_d_real_loss', discriminator_loss_dict['d_real_loss'])
        self.log('train_d_fake_loss', discriminator_loss_dict['d_fake_loss'])
        self.log('train_d_loss', discriminator_loss_dict['d_total_loss'], prog_bar=True)

        # Zero the gradients and perform backpropagation
        d_opt.zero_grad()
        self.manual_backward(discriminator_loss_dict['d_total_loss'])
        d_opt.step()

        self.log('train_loss', generator_loss_dict['g_total_loss'] + discriminator_loss_dict['d_total_loss'])

    def validation_step(self, batch):
        """
        Perform a validation step.

        Args:
            batch (tuple): A tuple containing the noisy and clean audio samples.

        This function performs a validation step for the SEGAN model. It first computes the intermediate step by calling the `_compute_intermediate_step` method. Then, it gets the targets for real and fake examples by calling the `_get_targets_for_real_and_fake_examples` method.

        After that, it calculates the generator loss by calling the `_calc_generator_loss` method and logs the generator loss. Similarly, it calculates the discriminator loss by calling the `_calc_discriminator_loss` method and logs the discriminator loss.
        """
        # Compute the intermediate step
        intermediate_step = self._compute_intermediate_step(batch, log_prefix=None)

        # Get the targets for real and fake examples
        valid, fake = self._get_targets_for_real_and_fake_examples(intermediate_step['batch_size'])

        #####################
        # Generator Metrics #
        #####################
        # Calculate the generator loss
        generator_loss_dict = self._calc_generator_loss(intermediate_step, valid)
        composite_metrics_dict = self._calc_generator_composite_metrics(intermediate_step)

        # Log the generator loss
        self.log('valid_g_loss', generator_loss_dict['g_loss'], sync_dist=True)
        self.log('valid_g_l1_loss', generator_loss_dict['g_l1_loss'], sync_dist=True)
        self.log('valid_g_total_loss', generator_loss_dict['g_total_loss'], prog_bar=True, sync_dist=True)

        if composite_metrics_dict is not None:
            self.log('valid_PESQ', composite_metrics_dict['PESQ'], prog_bar=True, sync_dist=True)
            self.log('valid_CSIG', composite_metrics_dict['CSIG'], prog_bar=True, sync_dist=True)
            self.log('valid_CBAK', composite_metrics_dict['CBAK'], prog_bar=True, sync_dist=True)
            self.log('valid_COVL', composite_metrics_dict['COVL'], prog_bar=True, sync_dist=True)
            self.log('valid_SSNR', composite_metrics_dict['SSNR'], prog_bar=True, sync_dist=True)

        #########################
        # Discriminator Metrics #
        #########################

        # Calculate the discriminator loss
        discriminator_loss_dict = self._calc_discriminator_loss(intermediate_step, valid, fake)

        # Log the discriminator loss
        self.log('valid_d_real_loss', discriminator_loss_dict['d_real_loss'], sync_dist=True)
        self.log('valid_d_fake_loss', discriminator_loss_dict['d_fake_loss'], sync_dist=True)
        self.log('valid_d_loss', discriminator_loss_dict['d_total_loss'], prog_bar=True, sync_dist=True)

        self.log('val_loss', generator_loss_dict['g_total_loss'] + discriminator_loss_dict['d_total_loss'],
                 sync_dist=True)

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
