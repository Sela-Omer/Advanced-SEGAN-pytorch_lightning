from typing import Optional, Callable

from torch import nn, Tensor


class TransposedResBlock1d(nn.Module):
    def __init__(
            self,
            inplanes: int,  # Number of input channels
            planes: int,  # Number of output channels
            stride: int = 2,  # Stride for the convolutional layer
            kernel_size: int = 31,  # Kernel size for the convolutional layer
            padding: int = 15,  # Padding for the convolutional layer
            dilation: int = 1,  # Dilation for the convolutional layer
            output_padding: int = 1,  # Output padding for the transposed convolutional layer
            prev_layer_inplanes: Optional[int] = None,  # Number of input channels in the previous layer, if any
            norm_layer: Optional[Callable[..., nn.Module]] = None,  # Normalization layer to use, if any
    ) -> None:
        """
        Initializes a TransposedResBlock1d instance.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int, optional): Stride for the transposed convolutional layer. Defaults to 2.
            kernel_size (int, optional): Kernel size for the transposed convolutional layer. Defaults to 31.
            padding (int, optional): Padding for the transposed convolutional layer. Defaults to 15.
            dilation (int, optional): Dilation for the transposed convolutional layer. Defaults to 1.
            output_padding (int, optional): Output padding for the transposed convolutional layer. Defaults to 1.
            prev_layer_inplanes (Optional[int], optional): Number of input channels in the previous layer, if any. Defaults to None.
            norm_layer (Optional[Callable[..., nn.Module]], optional): Normalization layer to use, if any. Defaults to None.
        """
        super().__init__()

        # If no normalization layer is provided, use BatchNorm1d as the default
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        # Define the transposed convolutional layer
        self.conv1 = nn.ConvTranspose1d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, output_padding=output_padding)
        # Define the normalization layer
        self.bn1 = norm_layer(planes)
        # Define the ReLU activation function
        self.relu = nn.ReLU(inplace=True)

        # If there is a previous layer, define a downsampling path
        if prev_layer_inplanes is not None:
            self.downsample = nn.ConvTranspose1d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                                 padding=padding, dilation=dilation,
                                                 output_padding=output_padding * 2)

    def forward(self, x: Tensor, prev_layer_input: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the TransposedResBlock1d.

        This function applies convolution, batch normalization, ReLU activation,
        and residual connection (if applicable) to the input tensor.

        Args:
            x (Tensor): Input tensor.
            prev_layer_input (Optional[Tensor], optional): Output from the previous layer. Defaults to None.

        Returns:
            Tensor: Output tensor.
        """
        # Apply convolution
        out = self.conv1(x)
        # Apply batch normalization
        out = self.bn1(out)
        # Apply ReLU activation
        out = self.relu(out)

        # If the TransposedResBlock1d has a downsampling layer (i.e., a previous layer exists),
        # downsample the previous layer's output and add it to the current output
        if hasattr(self, 'downsample'):
            # Downsample the previous layer's output
            prev_layer_input = self.downsample(prev_layer_input)
            # Add the downsampled output to the current output
            out += prev_layer_input

        return out
