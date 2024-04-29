from typing import Any, Tuple

import lightning as pl
import numpy as np
import torch
import torch.optim as optim
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from src.helper.metrics_helper import composite_metrics
from src.module.segan import SEGAN
from src.service.service import Service


class ImprovedSEGAN(SEGAN):
    def __init__(self, *args, noisy_mean=0, noisy_std=1, clean_mean=0, clean_std=1, **kwargs):
        """
        Initializes the ImprovedSEGAN model with the provided means and standard deviations.

        Args:
            *args: Variable length argument list.
            noisy_mean (float): The mean value for noisy audio.
            noisy_std (float): The standard deviation for noisy audio.
            clean_mean (float): The mean value for clean audio.
            clean_std (float): The standard deviation for clean audio.
            **kwargs: Arbitrary keyword arguments.

        """
        super(ImprovedSEGAN, self).__init__(*args, **kwargs)
        self.noisy_mean = noisy_mean
        self.noisy_std = noisy_std
        self.clean_mean = clean_mean
        self.clean_std = clean_std

        # normalize the l1 loss weighting by the standard deviation of the clean audio
        self.lambda_l1 *= self.clean_std

    def _normalize_batch(self, noisy: torch.Tensor, clean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize the noisy and clean audio batches.

        Args:
            noisy (torch.Tensor): The noisy audio batch.
            clean (torch.Tensor): The clean audio batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The normalized noisy and clean audio batches.
        """
        # Subtract the mean of noisy audio and divide by the standard deviation
        norm_noisy = (noisy.clone() - self.noisy_mean) / self.noisy_std

        # Subtract the mean of clean audio and divide by the standard deviation
        norm_clean = (clean.clone() - self.clean_mean) / self.clean_std

        return norm_noisy, norm_clean

    def _denormalize_batch(self, norm_noisy: torch.Tensor, norm_clean: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Denormalize the noisy and clean audio batches.

        Args:
            norm_noisy (torch.Tensor): The noisy audio batch.
            norm_clean (torch.Tensor): The clean audio batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The denormalized noisy and clean audio batches.
        """
        # Multiply by the standard deviation and add the mean
        denorm_noisy = norm_noisy.clone() * self.noisy_std + self.noisy_mean

        # Multiply by the standard deviation and add the mean
        denorm_clean = norm_clean.clone() * self.clean_std + self.clean_mean

        return denorm_noisy, denorm_clean

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
        norm_noisy, norm_clean = self._normalize_batch(noisy, clean)
        if z is None:
            # create thought vector
            z = torch.zeros([norm_noisy.size(0), 1024, 8]).to(device=norm_noisy.device)

        # Apply the generator neural network to the noisy input
        generated_clean = self.generator(norm_noisy, z.clone())[:, 0].unsqueeze(1)

        # Denormalize the generated clean audio
        _, generated_clean = self._denormalize_batch(norm_noisy, generated_clean)

        return generated_clean

    def _compute_intermediate_step(self, batch, log_prefix='train'):
        """
            Compute intermediate steps for the improved SEGAN model.
            Args:
                batch (Tuple[torch.Tensor, torch.Tensor]): The input batch containing denormalized noisy and clean audio.
                log_prefix (str, optional): The prefix for logging. Defaults to 'train'.
            Returns:
                dict: A dictionary containing various intermediate results.
                    - generator_fake_pred (torch.Tensor): The predicted scores from the generator for the generated and reference batches.
                    - denorm_generator_raw_pred (torch.Tensor): The denormalized raw predictions from the generator.
                    - discriminator_real_pred (torch.Tensor): The predicted scores from the discriminator for the clean and noisy audio.
                    - discriminator_fake_pred (torch.Tensor): The predicted scores from the discriminator for the generated and reference batches.
                    - norm_clean (torch.Tensor): The normalized clean audio.
                    - norm_noisy (torch.Tensor): The normalized noisy audio.
                    - denorm_clean (torch.Tensor): The denormalized clean audio.
                    - denorm_noisy (torch.Tensor): The denormalized noisy audio.
                    - batch_size (int): The size of the batch.
                    - norm_ref_batch (torch.Tensor): The normalized reference batch.
                    - denorm_ref_batch (torch.Tensor): The denormalized reference batch.
                    - norm_generated_ref_batch (torch.Tensor): The normalized reference batch generated by the generator.
                    - norm_generated_clean (torch.Tensor): The normalized clean audio generated by the generator.
        """
        # Unpack the batch
        denorm_noisy, denorm_clean = batch

        # Normalize the noisy and clean audio batches
        norm_noisy, norm_clean = self._normalize_batch(denorm_noisy, denorm_clean)

        # Create a detached copy of the reference batch
        denorm_ref_batch = self.ref_batch.clone().detach()
        denorm_ref_X, denorm_ref_y = denorm_ref_batch[:, 0].unsqueeze(1), denorm_ref_batch[:, 1].unsqueeze(1)

        # Normalize the reference batch
        norm_ref_noisy, norm_ref_clean = self._normalize_batch(denorm_ref_X, denorm_ref_y)
        norm_ref_batch = torch.cat((norm_ref_noisy, norm_ref_clean), dim=1)

        # create thought vector
        ref_z = self._make_z(tuple([int(denorm_ref_batch.size(0)), 1024, 8]), device=denorm_noisy.device)
        z = ref_z[:denorm_noisy.size(0)].clone()

        # Apply generator to reference batch
        norm_generated_ref_batch = self.generator(norm_ref_noisy, ref_z.clone()).detach()

        # Generate a cleaned version of the noisy audio
        if log_prefix is None:
            norm_generated_clean = self.generator(norm_noisy, z.clone())
        else:
            norm_generated_clean = self.generator(norm_noisy, z.clone(), log=self.log,
                                                  log_prefix=f'{log_prefix}_generator')

        ##########################
        # Compute Generator Step #
        ##########################
        # The generator's goal is to fool the discriminator
        generator_fake_pred = self.discriminator(norm_generated_clean, norm_generated_ref_batch)
        norm_generator_raw_pred = norm_generated_clean[:, 0].unsqueeze(1)

        # Denormalize the generator raw prediction
        _, denorm_generator_raw_pred = self._denormalize_batch(norm_noisy, norm_generator_raw_pred)

        ##############################
        # Compute Discriminator Step #
        ##############################
        # How well can the discriminator differentiate clean sounds?
        discriminator_real_pred = self.discriminator(torch.cat((norm_clean, norm_noisy), dim=1), norm_ref_batch)
        discriminator_fake_pred = self.discriminator(norm_generated_clean.detach(), norm_generated_ref_batch)

        return {
            'generator_fake_pred': generator_fake_pred,
            'norm_generator_raw_pred': norm_generator_raw_pred,
            'denorm_generator_raw_pred': denorm_generator_raw_pred,
            'discriminator_real_pred': discriminator_real_pred,
            'discriminator_fake_pred': discriminator_fake_pred,
            'norm_clean': norm_clean,
            'norm_noisy': norm_noisy,
            'denorm_clean': denorm_clean,
            'denorm_noisy': denorm_noisy,
            'batch_size': norm_noisy.size(0),
            'norm_ref_batch': norm_ref_batch,
            'denorm_ref_batch': denorm_ref_batch,
            'norm_generated_ref_batch': norm_generated_ref_batch,
            'norm_generated_clean': norm_generated_clean,
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
        norm_generator_raw_pred = intermediate_step['norm_generator_raw_pred']
        norm_clean = intermediate_step['norm_clean']

        # Calculate the generator loss
        g_loss = 0.5 * self.criterionGAN(generator_fake_pred, valid)

        # Calculate the L1 loss
        l1_loss = self.criterionL1(norm_generator_raw_pred, norm_clean)

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
        denorm_generator_raw_pred = intermediate_step['denorm_generator_raw_pred']
        denorm_clean = intermediate_step['denorm_clean']

        # Convert the clean and generator raw prediction tensors to numpy arrays
        denorm_clean_batch_arr = denorm_clean.detach().cpu().squeeze(1).numpy()
        denorm_generator_raw_pred_batch_arr = denorm_generator_raw_pred.detach().cpu().squeeze(1).numpy()

        comp_metrics = None
        count = 0
        for i in range(denorm_clean_batch_arr.shape[0]):
            clean_arr = denorm_clean_batch_arr[i]
            generator_raw_pred_arr = denorm_generator_raw_pred_batch_arr[i]
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
        norm_generated_clean = intermediate_step['norm_generated_clean']
        norm_generated_ref_batch = intermediate_step['norm_generated_ref_batch']
        norm_clean = intermediate_step['norm_clean']
        norm_noisy = intermediate_step['norm_noisy']
        norm_ref_batch = intermediate_step['norm_ref_batch']

        # Calculate the discriminator predictions for real and fake examples
        real_pred = self.discriminator(torch.cat((norm_clean, norm_noisy), dim=1), norm_ref_batch)
        fake_pred = self.discriminator(norm_generated_clean.detach(), norm_generated_ref_batch)

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

    def configure_optimizers(self):
        """
        This method configures the optimizers for the generator and discriminator.

        Returns:
            list: A list containing the optimizers for the generator and discriminator.
        """
        optimizer_gen = optim.Adam(
            self.generator.parameters(),
            lr=1e-4
        )
        optimizer_disc = optim.Adam(
            self.discriminator.parameters(),
            lr=1e-4
        )

        # # Create an RMSprop optimizer for the generator
        # optimizer_gen = optim.RMSprop(
        #     self.generator.parameters(),
        #     lr=self.lr_gen
        # )
        #
        # # Create an RMSprop optimizer for the discriminator
        # optimizer_disc = optim.RMSprop(
        #     self.discriminator.parameters(),
        #     lr=self.lr_disc
        # )

        return [optimizer_gen, optimizer_disc]
