import os
from typing import Union

import numpy as np
from PIL.Image import Image
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.contrib.keras.api.keras.preprocessing.image import img_to_array
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, InputLayer

from core.networks.abstract_network import _StyleTransferNetwork


class StyleTransferVGG19(_StyleTransferNetwork):

    def __init__(self, content_image: Image, style_image: Image, path: str = ''):
        super().__init__()

        # Get the content image's width and height.
        width, height = content_image.size

        # Resize the style image corresponding to the content image.
        style_image = style_image.resize((width, height))

        # Set the dimensions of the generated picture.
        self.img_nrows = height
        self.img_ncols = width

        # Create tensor representations of the given images.
        self._content_image = K.variable(self.preprocess_image(content_image))
        self._style_image = K.variable(self.preprocess_image(style_image))
        self.combination_image = K.placeholder((1, self.img_nrows, self.img_ncols, 3))

        # Combine the 3 images into a single tensor.
        input_tensor = K.concatenate([self._content_image,
                                      self._style_image,
                                      self.combination_image], axis=0)

        if path == '':
            # Build the VGG19 network, using pre-trained ImageNet weights.
            model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
        else:
            # Create the network, using the weights of the path's file.
            model = self.network(input_tensor, path)

        # Get the symbolic outputs of each "key" layer.
        self._outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

        self.content_features_layer = self._outputs_dict['block5_conv2']
        self.style_features_layers = [self._outputs_dict[layer_name] for layer_name in ['block1_conv1', 'block2_conv1',
                                                                                        'block3_conv1', 'block4_conv1',
                                                                                        'block5_conv1']]

    @staticmethod
    def network(input_tensor, weights_path: Union[None, str] = None) -> Sequential:
        """
        Defines a vgg19 network.

        :param input_tensor: the input tensor of the network.
        :param weights_path: a path to a trained vgg19 network's weights.

        :return: Keras Sequential Model.
        """
        # Create a Sequential model
        model = Sequential(name='vgg19')
        # Create an InputLayer using the input tensor.
        model.add(InputLayer(input_tensor=input_tensor))

        # Block 1
        model.add(Conv2D(64, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block1_conv1'))
        model.add(Conv2D(64, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block1_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        model.add(Conv2D(128, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block2_conv1'))
        model.add(Conv2D(128, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block2_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Conv2D(256, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block3_conv1'))
        model.add(Conv2D(256, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block3_conv2'))
        model.add(Conv2D(256, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block3_conv3'))
        model.add(Conv2D(256, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block3_conv4'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block4_conv1'))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block4_conv2'))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block4_conv3'))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block4_conv4'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block5_conv1'))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block5_conv2'))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block5_conv3'))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding='same',
                         name='block5_conv4'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        # Check if weights file exists.
        if not os.path.isfile(weights_path):
            raise FileNotFoundError('Network weights file {} does not exist.'.format(weights_path))

        # Load weights.
        model.load_weights(weights_path)

        return model

    @staticmethod
    def preprocess_image(image):
        """
        Resize and formats pictures into appropriate tensors for the VGG19 model.

        :param image: the image to preprocess.
        :return: the image preprocessed.
        """
        # Convert image to array.
        image = img_to_array(image)
        # Add extra dimension for the batches.
        image = np.expand_dims(image, axis=0)
        # image = tf.expand_dims(image, axis=0)

        # Preprocess the image and return it.
        return preprocess_input(image)

    def deprocess_image(self, x) -> np.ndarray:
        """
        Converts a tensor of VGG19 into a valid image.

        :param x: the tensor.
        :return: the image resulting from the tensor.
        """
        # Remove extra dimension for the batches.
        x = x.reshape((self.img_nrows, self.img_ncols, 3))

        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68

        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')

        return x
