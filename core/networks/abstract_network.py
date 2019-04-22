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
    def preprocess_image(image) -> dict:
        """
        Getter for the model's parameters and its values.

        :return: a dictionary containing the model's parameters and its values.
        """
        pass

    @abstractmethod
    def deprocess_image(self, x) -> np.ndarray:
        """
        Converts a tensor of Style Transfer Network into a valid image.

        :param x: the tensor.
        :return: the image resulting from the tensor.
        """
        pass
