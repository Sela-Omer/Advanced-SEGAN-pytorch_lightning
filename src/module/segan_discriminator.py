from typing import Tuple

import torch
from torch import nn

from src.module.virtual_batch_norm import VirtualBatchNorm1d


class SEGAN_Discriminator(nn.Module):
    """
    SEGAN Discriminator class.

    Args:
        batch_size (int): The batch size of the input data.
        time_dim_size (int): The dimensionality of the time dimension.
        n_out_features (int, optional): The number of output features. Defaults to 1.
        discriminator_dimensions (list, optional): The list of dimensions for the discriminator. Defaults to [2, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024].
        vbn_epsilon (float, optional): The epsilon value for the Virtual Batch Normalization. Defaults to 1e-5.

    Raises:
        AssertionError: If the time_dim_size is not an integer.
        AssertionError: If the n_out_features is not an integer.
        AssertionError: If the discriminator_dimensions is not a list.
        AssertionError: If the vbn_epsilon is not a float.

    Returns:
        nn.Module: The initialized SEGAN_Discriminator.

    Example:
        >>> discriminator = SEGAN_Discriminator(batch_size=400, time_dim_size=16384, n_out_features=1, discriminator_dimensions=[2, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024], vbn_epsilon=1e-5)
    """

    def __init__(self,
                 batch_size: int,
                 time_dim_size: int,
                 n_out_features: int = 1,
                 discriminator_dimensions: list = [2, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024],
                 vbn_epsilon: float = 1e-5):
        """
               Initialize the SEGAN_Discriminator object.
               Args:
                   batch_size (int): The size of the batch.
                   time_dim_size (int): The dimensionality of the time dimension.
                   n_out_features (int, optional): The number of output features. Defaults to 1.
                   discriminator_dimensions (list, optional): The dimensions of the discriminator layers. Defaults to [2, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024].
                   vbn_epsilon (float, optional): The epsilon value for the Virtual Batch Normalization. Defaults to 1e-5.
        """
        super(SEGAN_Discriminator, self).__init__()

        self.n_layers = len(discriminator_dimensions) - 1

        self.discriminator_dimensions = discriminator_dimensions
        self.vbn_epsilon = vbn_epsilon
        self.batch_size = batch_size
        self.time_dim_size = time_dim_size

        layers = []
        # Loop through the discriminator dimensions to create each block
        for i in range(len(discriminator_dimensions) - 1):
            block = self._make_discriminator_block(i)
            layers += block

        self.discriminator_layers = nn.ModuleList(layers)
        # Final convolutional layer
        self.final_conv = nn.Conv1d(discriminator_dimensions[-1], 1, kernel_size=1)
        # Fully connected layer
        self.fully_connected = nn.Linear(int(time_dim_size / (2 ** self.n_layers)), n_out_features)

    def _make_discriminator_block(self, index: int):
        """
        Creates a discriminator block for the SEGAN model.

        Args:
            index (int): The index of the block in the discriminator.

        Returns:
            List[nn.Module]: A list containing the convolutional layer, virtual batch normalization layer, and leaky ReLU activation layer.
        """
        # Get the input and output channels for the block
        in_channels, out_channels = self._discriminator_block_index_to_in_out_channels(index)

        # Create the convolutional layer
        conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=31, stride=2, padding=15)

        # Create the virtual batch normalization layer
        vbn_layer = VirtualBatchNorm1d(num_features=out_channels, batch_size=self.batch_size, epsilon=self.vbn_epsilon)

        # Create the leaky ReLU activation layer
        non_linear_layer = nn.LeakyReLU(0.3)

        # Create the block
        block = [
            conv_layer,
            vbn_layer,
            non_linear_layer
        ]

        # Return the block and its output
        return block

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

    def forward(self, x: torch.Tensor, ref_batch: torch.Tensor) -> torch.Tensor:
        """
        Apply forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input tensor.
            ref_batch (torch.Tensor): Reference input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the discriminator.
        """
        # Check input type and shape
        assert isinstance(x, torch.Tensor), f'x must be a torch.Tensor instead of {type(x)}'

        # Dictionary to store reference mean and mean squared values
        ref_stats_dict = {}

        # Apply layers to reference input
        for i, layer in enumerate(self.discriminator_layers):
            if isinstance(layer, VirtualBatchNorm1d):
                ref_batch, ref_mean, ref_mean_sq = layer(ref_batch)
                ref_stats_dict[i] = (ref_mean, ref_mean_sq)
            else:
                ref_batch = layer(ref_batch)

        # Apply layers to input with reference stats
        for i, layer in enumerate(self.discriminator_layers):
            if isinstance(layer, VirtualBatchNorm1d):
                ref_mean, ref_mean_sq = ref_stats_dict[i]
                x, _, _ = layer(x, ref_mean=ref_mean, ref_mean_sq=ref_mean_sq)
            else:
                x = layer(x)

        # Apply final convolutional layer
        x = self.final_conv(x)

        # Squeeze the tensor along the second dimension
        x = x.squeeze(1)

        # Apply fully connected layer
        x = self.fully_connected(x)

        return x
