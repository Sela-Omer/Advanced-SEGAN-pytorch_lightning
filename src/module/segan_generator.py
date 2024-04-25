from typing import Callable

import torch
from torch import nn
from torch.autograd import Variable


class SEGAN_Generator(nn.Module):
    """
    The SEGAN Generator model.

    Args:
        encoder_dimensions (List[int], optional): The dimensions of the encoder layers. Defaults to [1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024].

    Examples:
        >>> generator = SEGAN_Generator(encoder_dimensions=[1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024])
    """

    def __init__(self, encoder_dimensions=[1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]):
        """
        Initialize the SEGAN Generator model.

        Args:
            encoder_dimensions (List[int], optional): The dimensions of the encoder layers. Defaults to [1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024].
        """
        super().__init__()
        self.encoder_dimensions = encoder_dimensions
        self.num_layers = len(encoder_dimensions) - 1

        # Initialize the encoder and decoder layers
        # The encoder is a stack of convolutional layers
        self.encoder = nn.ModuleList([self._make_encoder_layer(index) for index in range(self.num_layers)])
        # The decoder is a stack of transposed convolutional layers
        self.decoder = nn.ModuleList([self._make_decoder_layer(index) for index in range(self.num_layers)])

    def _encoder_index_to_in_out_channels(self, index):
        """
        Get the input and output channels for a given encoder layer index.

        Args:
            index (int): The index of the encoder layer.

        Returns:
            tuple: A tuple containing the input and output channels.
        """
        # Get the input channels for the encoder layer
        in_channels = self.encoder_dimensions[index]

        # Get the output channels for the encoder layer
        out_channels = self.encoder_dimensions[index + 1]

        return in_channels, out_channels

    def _decoder_index_to_in_out_channels(self, index):
        """
        Get the input and output channels for a given decoder layer index.

        Args:
            index (int): The index of the decoder layer.

        Returns:
            tuple: A tuple containing the input and output channels.
        """
        # Calculate the decoder dimensions by doubling the encoder dimensions and reversing the order
        decoder_dimensions = [dim for dim in self.encoder_dimensions[::-1]]

        # Get the input channels for the decoder layer
        in_channels = decoder_dimensions[index] * 2

        # Get the output channels for the decoder layer
        out_channels = decoder_dimensions[index + 1]

        return in_channels, out_channels

    def _make_encoder_layer(self, index) -> nn.Module:
        """
        Creates a convolutional layer for the encoder.

        Parameters:
            index (int): The index of the layer.

        Returns:
            nn.Sequential: The created convolutional layer.
        """
        in_channels, out_channels = self._encoder_index_to_in_out_channels(index)
        return nn.Sequential(nn.Conv1d(in_channels, out_channels, 31, stride=2, padding=15), nn.PReLU())

    def _make_decoder_layer(self, index) -> nn.Module:
        """
        Create a decoder layer for the SEGAN generator.

        Parameters:
            index (int): The index of the layer.

        Returns:
            nn.Sequential: The created decoder layer.
        """
        # Get the input and output channels based on the index
        in_channels, out_channels = self._decoder_index_to_in_out_channels(index)

        # Create a sequential module with ConvTranspose1d and PReLU activation
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=31, stride=2, dilation=1, padding=15,
                               output_padding=1),
            nn.PReLU()
        )

    def _log(self, x: torch.Tensor, index: int, log: Callable[[str, torch.Tensor], None], prefix: str = "") -> None:
        """
        Log the mean and standard deviation of a tensor.

        Args:
            x (torch.Tensor): The tensor to log.
            index (int): The index of the tensor.
            log (Callable[[str, torch.Tensor], None]): The log function.
            prefix (str, optional): The prefix for the log key. Defaults to "".
        """
        # Return if log is None
        if log is None:
            return

        layer_index_s = f'_layer_{index}' if index != -1 else ''
        log_base_s = f'{prefix}{layer_index_s}_{tuple(x.shape)}'

        # Log the mean of the tensor
        log(f'{log_base_s}_mean', x.mean())

        # Log the standard deviation of the tensor
        log(f'{log_base_s}_std', x.std())

    def forward(self, x: torch.Tensor, z: torch.Tensor, log=None, log_prefix='generator'):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (Batch, Channels, Time).
            z (torch.Tensor): The random thought tensor of shape (Batch, Channels, Time).
            log (Callable[[str, torch.Tensor], None], optional): The logging function. Defaults to None.
            log_prefix (str, optional): The prefix for the log keys. Defaults to 'generator'.

        Returns:
            torch.Tensor: The output tensor of the forward pass.
        """
        # Check input tensor shape
        assert len(x.shape) == 3, "Input tensor must be 3D, Batch x Channels x Time"

        self._log(x, -1, log, prefix=f"{log_prefix}_input")

        # Encoding through all layers with skip connections
        skips = []  # List to store skip connections
        for i, layer in enumerate(self.encoder):
            skips.append(x)
            x = layer(x)
            self._log(x, i, log, prefix=f"{log_prefix}_encoder")

        # Concatenate the encoded tensor with a random tensor
        x = torch.cat([x, z], dim=1)

        # Reverse the skip connections list for easier indexing
        skips = skips[::-1]

        # Decoding through all layers with skip connections
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            self._log(x, i, log, prefix=f"{log_prefix}_decoder")
            x = torch.cat([x, skips[i]], dim=1)

        self._log(x, -1, log, prefix=f"{log_prefix}_output")

        return x
