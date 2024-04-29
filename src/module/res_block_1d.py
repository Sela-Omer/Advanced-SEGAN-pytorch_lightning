from typing import Optional, Callable

from torch import nn, Tensor


class ResBlock1d(nn.Module):
    """
    Residual block with 1D convolutional layers.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the convolutional layer. Defaults to 2.
        kernel_size (int, optional): Kernel size for the convolutional layer. Defaults to 31.
        padding (int, optional): Padding for the convolutional layer. Defaults to 15.
        prev_layer_inplanes (Optional[int], optional): Number of input channels in the previous layer, if any. Defaults to None.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Normalization layer to use, if any. Defaults to None.

    Examples:
        >>> res_block = ResBlock1d(inplanes=64, planes=64, stride=2, kernel_size=31, padding=15)
    """
    def __init__(
            self,
            inplanes: int,  # Number of input channels
            planes: int,  # Number of output channels
            stride: int = 2,  # Stride for the convolutional layer
            kernel_size: int = 31,  # Kernel size for the convolutional layer
            padding: int = 15,  # Padding for the convolutional layer
            prev_layer_inplanes: Optional[int] = None,  # Number of input channels in the previous layer, if any
            norm_layer: Optional[Callable[..., nn.Module]] = None,  # Normalization layer to use, if any
    ) -> None:
        """
        Initializes a ResBlock1d instance.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int, optional): Stride for the convolutional layer. Defaults to 2.
            kernel_size (int, optional): Kernel size for the convolutional layer. Defaults to 31.
            padding (int, optional): Padding for the convolutional layer. Defaults to 15.
            prev_layer_inplanes (Optional[int], optional): Number of input channels in the previous layer, if any. Defaults to None.
            norm_layer (Optional[Callable[..., nn.Module]], optional): Normalization layer to use, if any. Defaults to None.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d  # Use BatchNorm1d as the default normalization layer

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)  # Convolutional layer
        self.bn1 = norm_layer(planes)  # Normalization layer
        self.relu = nn.ReLU()  # ReLU activation function

        if prev_layer_inplanes is not None:
            self.downsample = nn.Conv1d(prev_layer_inplanes, planes, kernel_size=kernel_size, stride=stride*2,
                                        padding=padding)  # Downsampling layer

    def forward(self, x: Tensor, prev_layer_input: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the ResBlock1d.

        Args:
            x (Tensor): Input tensor.
            prev_layer_input (Optional[Tensor], optional): Output from the previous layer. Defaults to None.

        Returns:
            Tensor: Output tensor.
        """
        out = self.conv1(x)  # Convolution
        out = self.bn1(out)  # Batch normalization
        out = self.relu(out)  # ReLU activation

        if hasattr(self, 'downsample'):
            prev_layer_input = self.downsample(prev_layer_input)  # Downsampling
            out = out + prev_layer_input  # Residual connection

        return out
