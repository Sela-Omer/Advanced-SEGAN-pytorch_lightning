from typing import Tuple

import torch
from torch import nn

from src.module.virtual_batch_norm import VirtualBatchNorm1d


class SEGAN_Discriminator(nn.Module):
    """
    SEGAN Discriminator class.

    Args:
        ref_batch (torch.Tensor): The reference batch for the discriminator.
        n_out_features (int, optional): The number of output features. Defaults to 1.
        discriminator_dimensions (list, optional): The dimensions of the discriminator. Defaults to [2, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024].
        vbn_epsilon (float, optional): The epsilon value for VirtualBatchNorm1d. Defaults to 1e-5.

    Example:
        >>> discriminator = SEGAN_Discriminator(torch.rand((400,2,16384)), n_out_features=1, discriminator_dimensions=[2, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024], vbn_epsilon=1e-5)
    """

    def __init__(self, ref_batch: torch.Tensor,
                 n_out_features: int = 1,
                 discriminator_dimensions: list = [2, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024],
                 vbn_epsilon: float = 1e-5):
        """
        Initialize the SEGAN Discriminator.

        Args:
            ref_batch (torch.Tensor): The reference batch for the discriminator.
            n_out_features (int, optional): The number of output features. Defaults to 1.
            discriminator_dimensions (list, optional): The dimensions of the discriminator. Defaults to [2, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024].
            vbn_epsilon (float, optional): The epsilon value for VirtualBatchNorm1d. Defaults to 1e-5.
        """
        super(SEGAN_Discriminator, self).__init__()

        # Ensure the reference batch is a torch.Tensor
        assert isinstance(ref_batch, torch.Tensor), f'ref_batch must be a torch.Tensor instead of {type(ref_batch)}'
        self.ref_shape = ref_batch.shape
        # Ensure the reference batch is a 3D tensor (batch x features x time)
        assert len(
            self.ref_shape) == 3, f'ref_batch must be a 3D tensor (batch x features x time) instead of {self.ref_shape}'
        self.ref_time_size = int(self.ref_shape[2])
        self.n_layers = len(discriminator_dimensions) - 1

        self.discriminator_dimensions = discriminator_dimensions
        self.vbn_epsilon = vbn_epsilon

        layers = []
        running_vbn_ref_batch = ref_batch
        # Loop through the discriminator dimensions to create each block
        for i in range(len(discriminator_dimensions) - 1):
            block, running_vbn_ref_batch = self._make_discriminator_block(i, running_vbn_ref_batch)
            layers.append(block)

        self.discriminator_layers = nn.Sequential(*layers)
        # Final convolutional layer
        self.final_conv = nn.Conv1d(discriminator_dimensions[-1], 1, kernel_size=1)
        # Fully connected layer
        self.fully_connected = nn.Linear(int(self.ref_time_size / (2**self.n_layers)), n_out_features)

    def _make_discriminator_block(self, index: int, running_vbn_ref_batch: torch.Tensor):
        """
        This function creates a discriminator block for a SEGAN model.

        Args:
            index (int): The index of the block in the discriminator.
            running_vbn_ref_batch (torch.Tensor): The reference batch for VirtualBatchNorm1d.

        Returns:
            Tuple[nn.Sequential, torch.Tensor]: A tuple containing the block and the output of the block.
        """
        # Get the input and output channels for the block
        in_channels, out_channels = self._discriminator_block_index_to_in_out_channels(index)

        assert in_channels == running_vbn_ref_batch.shape[1], f'at index {index}, ref in_channels must be {running_vbn_ref_batch.shape[1]} equal to {out_channels}'

        conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=31, stride=2, padding=15)  # Convolutional layer
        running_vbn_ref_batch = conv_layer.forward(running_vbn_ref_batch)

        vbn_layer = VirtualBatchNorm1d(running_vbn_ref_batch, epsilon=self.vbn_epsilon)  # Virtual Batch Normalization layer
        running_vbn_ref_batch = vbn_layer.forward(running_vbn_ref_batch)

        non_linear_layer = nn.LeakyReLU(0.3)  # Leaky ReLU activation layer
        running_vbn_ref_batch = non_linear_layer.forward(running_vbn_ref_batch)

        # Create the block
        block = nn.Sequential(
            conv_layer,
            vbn_layer,
            non_linear_layer
        )

        # Return the block and its output
        return block, running_vbn_ref_batch

    def _discriminator_block_index_to_in_out_channels(self, index: int) -> Tuple[int, int]:
        """
        Given an index, returns the input and output channels for a discriminator block.

        Args:
            index (int): The index of the block in the discriminator.

        Returns:
            Tuple[int, int]: A tuple containing the input and output channels.
        """
        # Get the input and output channels for the block
        in_channels = self.discriminator_dimensions[index]
        out_channels = self.discriminator_dimensions[index + 1]

        return in_channels, out_channels

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the SEGAN Discriminator network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        # Check input type and shape
        assert isinstance(x, torch.Tensor), f'x must be a torch.Tensor instead of {type(x)}'
        assert tuple(x.shape[1:]) == tuple(self.ref_shape[1:]), f'x must have shape {self.ref_shape} instead of {x.shape}'

        # Apply discriminator layers
        x = self.discriminator_layers(x)

        # Apply final convolutional layer
        x = self.final_conv(x)

        # Squeeze the tensor along the second dimension
        x = x.squeeze(1)

        # Apply fully connected layer
        x = self.fully_connected(x)

        return x
