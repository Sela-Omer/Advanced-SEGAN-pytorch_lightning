import torch
import torch.nn as nn


class VirtualBatchNorm1d(nn.Module):
    def __init__(self, num_features: int, batch_size: int, epsilon: float = 1e-5):
        super(VirtualBatchNorm1d, self).__init__()

        # Assert that num_features is an integer
        assert isinstance(num_features, int), f'num_features must be an integer instead of {type(num_features)}'

        # Assert that batch_size is an integer
        assert isinstance(batch_size, int), f'batch_size must be an integer instead of {type(batch_size)}'

        # Assert that epsilon is a float
        assert isinstance(epsilon, float), f'epsilon must be a float instead of {type(epsilon)}'

        self.batch_size = batch_size

        # Set the epsilon value
        self.epsilon = epsilon

        # Initialize the gamma and beta parameters
        self.gamma = self._init_reg_param(torch.normal(mean=torch.ones(1, num_features, 1), std=0.02), 'gamma')
        self.beta = self._init_reg_param(torch.zeros(1, num_features, 1), 'beta')

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

    def forward(self, x: torch.Tensor, ref_mean: torch.Tensor = None, ref_mean_sq: torch.Tensor = None):
        # Check if x is a torch.Tensor
        assert isinstance(x, torch.Tensor), f'input must be a torch.Tensor instead of {type(x)}'

        # Calculate the new mean and mean squared
        new_mean = torch.mean(x, dim=(0, 1), keepdim=True)
        new_mean_sq = torch.mean(torch.square(x), dim=[0, 1], keepdim=True)

        if ref_mean is None or ref_mean_sq is None:
            mean = new_mean.clone().detach()
            mean_sq = new_mean_sq.clone().detach()
        else:
            # Calculate the new and old coefficients
            new_coeff = 1. / (self.batch_size + 1.)
            old_coeff = 1. - new_coeff

            # Calculate the overall mean and mean squared
            mean = new_coeff * new_mean + old_coeff * ref_mean
            mean_sq = new_coeff * new_mean_sq + old_coeff * ref_mean_sq

        # Normalize the input tensor
        out = self._normalize(x, mean, mean_sq)

        return out, mean, mean_sq
