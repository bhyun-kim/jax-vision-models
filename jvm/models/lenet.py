from flax import linen as nn

from jvm.registry import ModelRegistry


@ModelRegistry.register()
class LeNet5(nn.Module):
    """LeNet5 <https://ieeexplore.ieee.org/document/726791>.    
    """

    @nn.compact
    def __call__(self, x):
        x = LeNet5_Backbone()(x)
        x = LeNet5_Classifier()(x)
        return x


class LeNet5_Backbone(nn.Module):

    @nn.compact
    def __call__(self, x, train: bool = False):

        x = nn.Conv(features=6, kernel_size=(5, 5), use_bias=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(5, 5), use_bias=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        return x


class LeNet5_Classifier(nn.Module):

    @nn.compact
    def __call__(self, x, train: bool = False):

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(120, use_bias=False)(x)
        x = nn.relu(x)
        x = nn.Dense(84, use_bias=False)(x)
        x = nn.relu(x)
        x = nn.Dense(10, use_bias=False)(x)

        return x
