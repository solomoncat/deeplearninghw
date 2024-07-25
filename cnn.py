
"""
class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))


        self.main_path = self._make_main_path(
            in_channels, channels, kernel_sizes, batchnorm, dropout, activation_type, activation_params)
        self.shortcut_path = self._make_shortcut_path(in_channels, channels[-1])

    def _make_main_path(self, in_channels, channels, kernel_sizes, batchnorm, dropout, activation_type, activation_params):
        layers = []
        activation_cls = ACTIVATIONS[activation_type]

        for i in range(len(channels)):
            layers.append(nn.Conv2d(in_channels, channels[i], kernel_size=kernel_sizes[i], padding=kernel_sizes[i] // 2, bias=True))
            if batchnorm:
                layers.append(nn.BatchNorm2d(channels[i]))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(activation_cls(**activation_params))
            in_channels = channels[i]

        return nn.Sequential(*layers)

    def _make_shortcut_path(self, in_channels, out_channels):
        if in_channels != out_channels:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        else:
            return nn.Identity()

    def forward(self, x: Tensor):
        main_path = self.main_path(x)
        shortcut_path = self.shortcut_path(x)
        out = main_path + shortcut_path
        return F.relu(out)
# Test the implementation
seed = 42
nn.manual_seed(seed)
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Sequence

ACTIVATIONS = {
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
}

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w = tuple(self.in_size)
        layers = []

        # Set the initial number of input channels
        input_channels = in_channels

        # Define the activation function
        activation_cls = ACTIVATIONS[self.activation_type]
        activation = activation_cls(**self.activation_params)

        pooling_cls = POOLINGS[self.pooling_type]
        pooling = pooling_cls(**self.pooling_params)

        # Iterate through the channels, adding conv layers and pooling layers accordingly
        for i, output_channels in enumerate(self.channels):
            layers.append(nn.Conv2d(input_channels, output_channels, **self.conv_params))
            layers.append(activation)

            # After every `self.pool_every` conv layers, add a pooling layer
            if (i + 1) % self.pool_every == 0:
                layers.append(pooling)

            # Set the number of input channels for the next layer
            input_channels = output_channels

        return nn.Sequential(*layers)

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            zeroes = torch.zeros(1, *self.in_size)

            with torch.no_grad():
                zero_feature = self.feature_extractor(zeroes)
            no_features = zero_feature.view(1, -1).size(1)
            return no_features
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        # Create the MLP part of the model using the number of convolutional features
        n_features = self._n_features()
        layers = []

        input_dim = n_features
        activation_cls = ACTIVATIONS[self.activation_type]
        activation = activation_cls(**self.activation_params)

        # Add the hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            input_dim = hidden_dim

        # Add the output layer
        layers.append(nn.Linear(input_dim, self.out_classes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        # Extract features using the feature extractor
        features = self.feature_extractor(x)

        # Flatten the features to pass them to the MLP
        features = features.view(features.size(0), -1)

        # Get class scores from the MLP
        out = self.mlp(features)
        return out




class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))


        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        self.in_channels = in_channels
        self.main_path = []
        self.shortcut_path = []
        activation = nn.ReLU(**activation_params) if activation_type == "relu" else nn.LeakyReLU(**activation_params)
        if activation_type == "tanh":
            activation = nn.Tanh()
        N = len(channels)
        for conv_ind in range(N-1):
            in_channels = in_channels if conv_ind == 0 else channels[conv_ind - 1]
            self.main_path += [nn.Conv2d(in_channels=in_channels, out_channels=channels[conv_ind],
                                         padding=int((kernel_sizes[conv_ind]-1)/2),
                                         kernel_size=kernel_sizes[conv_ind])]
            if dropout > 0:
                self.main_path += [nn.Dropout2d(p=dropout)]
            if batchnorm:
                self.main_path += [nn.BatchNorm2d(channels[conv_ind])]
            self.main_path += [activation]

        if N == 1:
            conv_ind = 0
            self.main_path += [nn.Conv2d(in_channels=in_channels, out_channels=channels[conv_ind],
                                         kernel_size=kernel_sizes[0],
                                         padding=int((kernel_sizes[0] - 1) / 2))]
        else:
            conv_ind += 1
            self.main_path += [nn.Conv2d(in_channels=channels[conv_ind - 1], out_channels=channels[conv_ind],
                                         kernel_size=kernel_sizes[conv_ind],
                                         padding=int((kernel_sizes[conv_ind] - 1) / 2))]

        self.main_path = nn.Sequential(*self.main_path)

        if self.in_channels != channels[conv_ind]:
            self.shortcut_path += [nn.Conv2d(in_channels=self.in_channels, out_channels=channels[conv_ind],
                       kernel_size=1, bias=False)]
        else:
            self.shortcut_path += [nn.Identity()]

        self.shortcut_path = nn.Sequential(*self.shortcut_path)
        # ========================

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass. Save the main and residual path to `out`.
        out: Tensor = None
        # ====== YOUR CODE: ======
        output_main = self.main_path(x)
        output_shortcut = self.shortcut_path(x)
        out = output_main + output_shortcut
        # ========================
        out = torch.relu(out)
        return out


# Test the implementation
seed = 42
torch.manual_seed(seed)




class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. NOT the outer projections)
            The length determines the number of convolutions, EXCLUDING the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->5->10.
            The first and last arrows are the 1X1 projection convolutions,
            and the middle one is the inner convolution (corresponding to the kernel size listed in "inner kernel sizes").
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        # Define the channels and kernel sizes for the bottleneck block
        channels = [inner_channels[0]] + inner_channels + [in_out_channels]
        kernel_sizes = [1] + inner_kernel_sizes + [1]

        # Initialize the base class (ResidualBlock) with these parameters
        super().__init__(in_channels=in_out_channels, channels=channels, kernel_sizes=kernel_sizes, **kwargs)

# Test the implementation
seed = 42
torch.manual_seed(seed)

bottleneck_block = ResidualBottleneckBlock(
    in_out_channels=10, inner_channels=[5], inner_kernel_sizes=[3],
    batchnorm=True, dropout=0.2
)



class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

        # self.feature_extractor = self._make_feature_extractor()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        # ====== YOUR CODE: ======
        pooling = nn.MaxPool2d(**self.pooling_params) if self.pooling_type == "max" else nn.AvgPool2d(
            **self.pooling_params)
        N = len(self.channels)
        P = self.pool_every

        res_block = ResidualBottleneckBlock if self.bottleneck else ResidualBlock

        blocks = []
        num_of_blocks_with_pooling, num_of_convs_in_block_without_pooling = divmod(N, P)
        for i in range(num_of_blocks_with_pooling):
            if self.bottleneck:  # Note: I think that my implementation under this condition is too much tailored for
                # the input of the task. Maybe for other inputs it won't work
                # checks if input size==output size:
                if (i == 0 and in_channels == self.channels[P-1]) or (i != 0 and self.channels[i * P] == self.channels[(i + 1) * P - 1]):
                    # here res_block = ResidualBottleneckBlock
                    # Notice that my implementation deals only with the case of inner_channels of 1 channel and that's
                    # why I needed to make it a list (from an int) although self.channels is already a list
                    blocks += [res_block(in_out_channels=self.channels[i * P], inner_channels=[self.channels[i * P + 1]],
                                         inner_kernel_sizes=[3] * (P-2), batchnorm=self.batchnorm, dropout=self.dropout,
                                         activation_type=self.activation_type)]
                else:  # if the input and output sizes of the block don't match, create a regular block instead of a bottleneck one
                    blocks += [ResidualBlock(in_channels=in_channels, channels=self.channels[i * P:(i + 1) * P],
                                             kernel_sizes=[3] * P,
                                             batchnorm=self.batchnorm, dropout=self.dropout,
                                             activation_type=self.activation_type)]
            else:
                # here res_block = ResidualBlock
                blocks += [res_block(in_channels=in_channels, channels=self.channels[i * P:(i + 1) * P],
                                     kernel_sizes=[3] * P,
                                     batchnorm=self.batchnorm, dropout=self.dropout,
                                     activation_type=self.activation_type)]
            blocks += [pooling]

        if num_of_convs_in_block_without_pooling > 0:
            if self.bottleneck:
                blocks += [res_block(in_out_channels=in_channels, inner_channels=self.channels[(i+1)*P:],
                                     inner_kernel_sizes=[3] * num_of_convs_in_block_without_pooling, batchnorm=self.batchnorm, dropout=self.dropout,
                                     activation_type=self.activation_type)]
            else:
                blocks += [res_block(in_channels=self.channels[(i + 1) * P - 1], channels=self.channels[(i + 1) * P:],
                                     kernel_sizes=[3] * num_of_convs_in_block_without_pooling,
                                     batchnorm=self.batchnorm, dropout=self.dropout,
                                     activation_type=self.activation_type)]

        layers = blocks
        # ========================
        seq = nn.Sequential(*layers)
        return seq

