from typing import Callable

import torch
from torch import nn
from torch.autograd import Variable

from src.module.res_block_1d import ResBlock1d
from src.module.segan_generator import SEGAN_Generator
from src.module.transposed_res_block_1d import TransposedResBlock1d


class SEGAN_Residual_BN_Generator(SEGAN_Generator):
    """
    The SEGAN Residual BN Generator model.

    Args:
        encoder_dimensions (List[int], optional): The dimensions of the encoder layers. Defaults to [1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024].

    Examples:
        >>> generator = SEGAN_Residual_BN_Generator(encoder_dimensions=[1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024])
    """

    def _make_encoder_layer(self, index) -> nn.Module:
        """
        Create an encoder layer using ResBlock1d.

        Parameters:
            index (int): The index of the layer.

        Returns:
            nn.Module: The created encoder layer.
        """
        # Get the input and output channels for the encoder layer
        in_channels, out_channels = self._encoder_index_to_in_out_channels(index)

        # Get the input channels for the previous layer if available
        prev_in_channels, _ = self._encoder_index_to_in_out_channels(index - 1) if index > 0 else (None, None)

        # Create and return the ResBlock1d encoder layer
        return ResBlock1d(inplanes=in_channels, planes=out_channels, prev_layer_inplanes=prev_in_channels, stride=2,
                          kernel_size=31, padding=15)

    def _make_decoder_layer(self, index) -> nn.Module:
        """
        Create a decoder layer for the SEGAN Residual BN Generator.

        Parameters:
            index (int): The index of the layer.

        Returns:
            nn.Module: The created decoder layer.

        This method creates a decoder layer for the SEGAN Residual BN Generator. It first gets the input and output channels
        based on the index. Then it gets the input channels for the previous layer if available. If this is the last decoder
        layer, it returns a ConvTranspose1d layer with batch normalization, ReLU activation, and a final Conv1d layer.
        Otherwise, it returns a TransposedResBlock1d layer.
        """
        # Get the input and output channels based on the index
        in_channels, out_channels = self._decoder_index_to_in_out_channels(index)

        # Get the input channels for the previous layer if available
        prev_in_channels, _ = self._decoder_index_to_in_out_channels(index - 1) if index > 0 else (None, None)

        # If this is the last decoder layer, return a ConvTranspose1d layer with batch normalization, ReLU activation, and a final Conv1d layer
        if index == self.num_layers - 1:
            return nn.ModuleList([
                TransposedResBlock1d(inplanes=in_channels, planes=out_channels,
                                     stride=2, kernel_size=31, padding=15,
                                     dilation=1, output_padding=1,
                                     prev_layer_inplanes=prev_in_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 31, stride=1, padding=15),
            ])

        # Otherwise, return a TransposedResBlock1d layer
        return TransposedResBlock1d(inplanes=in_channels, planes=out_channels,
                                    stride=2, kernel_size=31, padding=15,
                                    dilation=1, output_padding=1,
                                    prev_layer_inplanes=prev_in_channels)

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
        prev_layer_input = None
        for i, layer in enumerate(self.encoder):
            skips.append(x)
            nxt_x = layer(x, prev_layer_input=prev_layer_input)
            prev_layer_input = x
            x = nxt_x
            self._log(x, i, log, prefix=f"{log_prefix}_encoder")

        # Concatenate the encoded tensor with a random tensor
        x = torch.cat([x, z], dim=1)

        # Reverse the skip connections list for easier indexing
        skips = skips[::-1]
        prev_layer_input = None

        # Decoding through all layers with skip connections
        for i, layer in enumerate(self.decoder):
            if i == self.num_layers - 1:
                x = layer[0](x, prev_layer_input=prev_layer_input)
                for sublayer in layer[1:]:
                    x = sublayer(x)
            else:
                nxt_x = layer(x, prev_layer_input=prev_layer_input)
                prev_layer_input = x
                x = nxt_x
            self._log(x, i, log, prefix=f"{log_prefix}_decoder")
            x = torch.cat([x, skips[i]], dim=1)

        self._log(x, -1, log, prefix=f"{log_prefix}_output")

        return x
