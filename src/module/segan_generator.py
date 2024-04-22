import torch
from torch import nn


class SEGAN_Generator(nn.Module):
    """
    The SEGAN Generator model.

    Args:
        encoder_dimensions (List[int], optional): The dimensions of the encoder layers. Defaults to [1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024].

    Examples:
        >>> generator = SEGAN_Generator()
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
        z = torch.normal(mean=torch.full(shape, mean), std=torch.full(shape, std))
        return z.to(device if device else torch.device("cpu"))

    def forward(self, x):
        """
        Apply the SEGAN generator to an input tensor.

        Args:
            x (torch.Tensor): The input tensor. It should be 3D, Batch x Channels x Time.

        Returns:
            torch.Tensor: The output tensor after applying the SEGAN generator.

        Raises:
            AssertionError: If the input tensor is not 3D or if the number of channels does not match the encoder dimensions.
        """
        # Check input tensor shape
        assert len(x.shape) == 3, "Input tensor must be 3D, Batch x Channels x Time"
        assert x.shape[1] == self.encoder_dimensions[0], (
            f"Input tensor must have the same number of channels as the encoder dimensions. "
            f"Instead got input channels {x.shape[1]} and encoder dimensions {self.encoder_dimensions[0]}"
        )

        # Encoding through all layers with skip connections
        skips = []  # List to store skip connections
        for i, layer in enumerate(self.encoder):
            skips.append(x)
            x = layer(x)


        # Concatenate the encoded tensor with a random tensor
        x = torch.cat([x, self._make_z(x.shape, device=x.device)], dim=1)

        # Reverse the skip connections list for easier indexing
        skips = skips[::-1]

        # Decoding through all layers with skip connections
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            x = torch.cat([x, skips[i]], dim=1)

        return x