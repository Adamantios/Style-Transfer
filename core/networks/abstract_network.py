from abc import abstractmethod, ABC
from typing import Union

import numpy as np
from tensorflow.python.keras import Sequential


class _StyleTransferNetwork(ABC):
    def __init__(self):
        pass

    @staticmethod
    def network(input_tensor, weights_path: Union[None, str] = None) -> Sequential:
        """
        Defines a Style Transfer Network.

        :param input_tensor: the input tensor of the network.
        :param weights_path: a path to a trained cifar-10 network's weights.

        :return: Keras Sequential Model.
        """
        pass

    @staticmethod
    def preprocess_image(image) -> np.ndarray:
        """
        Resize and formats pictures into appropriate tensors for the style transfer network.

        :param image: the image to preprocess.

        :return: the image as a numpy array, preprocessed.
        """
        pass

    @staticmethod
    def deprocess_image(x, shape) -> np.ndarray:
        """
        Converts a tensor of Style Transfer Network into a valid image.

        :param x: the tensor.
        :param shape: The shape of the resulting image.

        :return: the image resulting from the tensor.
        """
        pass
