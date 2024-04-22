import torch
import torch.nn as nn


class VirtualBatchNorm1d(nn.Module):
    """
    Virtual Batch Normalization 1D module.

    Args:
        ref_batch (torch.Tensor): Reference batch tensor.
        epsilon (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
    """

    def __init__(self, ref_batch: torch.Tensor, epsilon: float = 1e-5):
        """
        Initializes the VirtualBatchNorm1d module.

        Args:
            ref_batch (torch.Tensor): Reference batch tensor.
            epsilon (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.

        Raises:
            AssertionError: If epsilon is not a float or if ref_batch is not a torch.Tensor.
            AssertionError: If ref_batch is not a 3D tensor.

        """
        super().__init__()

        # Assert that epsilon is a float
        assert isinstance(epsilon, float), f'epsilon must be a float instead of {type(epsilon)}'

        # Assert that ref_batch is a torch.Tensor
        assert isinstance(ref_batch, torch.Tensor), f'ref_batch must be a torch.Tensor instead of {type(ref_batch)}'

        # Get the shape of the reference batch tensor
        self.ref_shape = ref_batch.shape

        # Assert that ref_batch is a 3D tensor
        assert len(
            self.ref_shape) == 3, f'ref_batch must be a 3D tensor (batch x features x time) instead of {self.ref_shape}'

        # Get the number of features from the reference batch tensor
        self.num_features = int(self.ref_shape[1])

        # Set the epsilon value
        self.epsilon = epsilon

        # Calculate the mean and mean square of the reference batch tensor
        self.ref_mean = self._init_reg_param(ref_batch.mean(dim=(0, 1), keepdim=True), 'ref_mean')
        self.ref_mean_sq = self._init_reg_param((ref_batch ** 2).mean(dim=(0, 1), keepdim=True), 'ref_mean_sq')

        # Get the batch size from the reference batch tensor
        self.batch_size = int(self.ref_shape[0])

        # Initialize the gamma and beta parameters
        self.gamma = self._init_reg_param(torch.normal(mean=torch.ones(1, self.num_features, 1), std=0.02), 'gamma')
        self.beta = self._init_reg_param(torch.zeros(1, self.num_features, 1), 'beta')

    def _init_reg_param(self, tensor: torch.Tensor, name: str) -> nn.Parameter:
        """
        Initializes a new parameter with the given tensor and name.

        Args:
            tensor (torch.Tensor): The tensor to initialize the parameter with.
            name (str): The name of the parameter.

        Returns:
            nn.Parameter: The initialized parameter.

        Raises:
            AssertionError: If the tensor is not a torch.Tensor.
            AssertionError: If the name is not a string.

        """
        # Check if the tensor is a torch.Tensor
        assert isinstance(tensor, torch.Tensor), f'{name} must be a torch.Tensor instead of {type(tensor)}'

        # Check if the name is a string
        assert isinstance(name, str), f'{name} must be a string instead of {type(name)}'

        # Initialize the parameter with the given tensor
        param = nn.Parameter(tensor)

        # Register the parameter with the given name
        self.register_parameter(name, param)

        # Return the initialized parameter
        return param

    def _normalize(self, x: torch.Tensor, mean: torch.Tensor, mean_sq: torch.Tensor):
        """
        Normalizes the input tensor using the provided mean and mean_sq tensors.

        Args:
            x (torch.Tensor): The input tensor to normalize.
            mean (torch.Tensor): The mean tensor.
            mean_sq (torch.Tensor): The mean square tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        Raises:
            AssertionError: If the shape of x is not (batch, features, time).
            AssertionError: If mean_sq is not a torch.Tensor.
            AssertionError: If mean is not a torch.Tensor.
        """
        # Check if the shape of x is (batch, features, time)
        shape = x.shape
        assert len(shape) == 3, f'x must be a 3D tensor (batch x features x time) instead of {shape}'

        # Check if mean_sq and mean are torch.Tensors
        assert isinstance(mean_sq, torch.Tensor), f'mean_sq must be a torch.Tensor instead of {type(mean_sq)}'
        assert isinstance(mean, torch.Tensor), f'mean must be a torch.Tensor instead of {type(mean)}'

        # Calculate the standard deviation
        std = torch.sqrt(self.epsilon + mean_sq - torch.square(mean))

        # Subtract the mean and divide by the standard deviation
        out = x - mean
        out = out / std

        # Scale by gamma and add beta
        out = out * self.gamma
        out = out + self.beta

        return out

    def forward(self, x: torch.Tensor):
        """
        Applies the forward pass of the virtual batch normalization layer to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: The output tensor after applying the virtual batch normalization layer.

        Raises:
            AssertionError: If the input is not a torch.Tensor.
            AssertionError: If the shape of the input tensor does not match the reference shape.
        """
        # Check if x is a torch.Tensor
        assert isinstance(x, torch.Tensor), f'input must be a torch.Tensor instead of {type(x)}'

        # Check if the shape of x matches the reference shape
        shape = x.shape
        assert tuple(shape) == tuple(self.ref_shape), f'input must have shape {self.ref_shape} instead of {shape}'

        # Calculate the new and old coefficients
        new_coeff = 1. / (self.batch_size + 1.)
        old_coeff = 1. - new_coeff

        # Calculate the new mean and mean squared
        new_mean = torch.mean(x, dim=(0, 1), keepdim=True)
        new_mean_sq = torch.mean(torch.square(x), dim=[0, 1], keepdim=True)

        # Calculate the overall mean and mean squared
        mean = new_coeff * new_mean + old_coeff * self.ref_mean
        mean_sq = new_coeff * new_mean_sq + old_coeff * self.ref_mean_sq

        # Normalize the input tensor
        out = self._normalize(x, mean, mean_sq)

        return out
