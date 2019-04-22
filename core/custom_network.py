import os
from typing import Union

import numpy as np
from PIL.Image import Image
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.preprocessing.image import img_to_array
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, InputLayer, BatchNormalization, Dropout, \
    Dense, Flatten
from tensorflow.python.keras.regularizers import l2


class StyleTransferCustom(object):

    def __init__(self, content_image: Image, style_image: Image, path: str):
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

        # Create the network, using the weights of the path's file.
        model = self.network(input_tensor, path)

        # Get the symbolic outputs of each "key" layer.
        self._outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

        self.content_features_layer = self._outputs_dict['block3_conv2']
        self.style_features_layers = [self._outputs_dict[layer_name] for layer_name in ['block1_conv1',
                                                                                        'block2_conv1',
                                                                                        'block3_conv1']]

    @staticmethod
    def network(input_tensor, weights_path: Union[None, str] = None):
        # Create a Sequential model.
        model = Sequential(name='custom_cifar-10')
        # Create an InputLayer using the input tensor.
        model.add(InputLayer(input_tensor=input_tensor))
        # Define a weight decay for the regularisation.
        weight_decay = 1e-4

        # Block1
        model.add(Conv2D(32, (3, 3), padding='same', activation='elu', name='block1_conv1',
                         kernel_regularizer=l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same', activation='elu', name='block1_conv2',
                         kernel_regularizer=l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool'))
        model.add(Dropout(0.2))

        # Block2
        model.add(Conv2D(64, (3, 3), padding='same', activation='elu', name='block2_conv1',
                         kernel_regularizer=l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding='same', activation='elu', name='block2_conv2',
                         kernel_regularizer=l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), name='block2_pool'))
        model.add(Dropout(0.3))

        # Block3
        model.add(Conv2D(128, (3, 3), padding='same', activation='elu', name='block3_conv1',
                         kernel_regularizer=l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding='same', activation='elu', name='block3_conv2',
                         kernel_regularizer=l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), name='block3_pool'))

        if weights_path is None:
            # Add top layer.
            model.add(Dropout(0.4))
            model.add(Flatten())
            model.add(Dense(10, activation='softmax'))
        else:
            if not os.path.isfile(weights_path):
                raise FileNotFoundError('Network weights file {} does not exist.'.format(weights_path))
            # Load weights.
            model.load_weights(weights_path)

        return model

    @staticmethod
    def preprocess_image(image):
        """
        Resize and formats pictures into appropriate tensors for the custom-cifar10 model.

        :param image: the image to preprocess.
        :return: the image preprocessed.
        """
        # Convert image to array.
        image = img_to_array(image)
        # Add extra dimension for the batches.
        image = np.expand_dims(image, axis=0)

        # Zero-center image.
        image[:, :, 0] -= 125.30691805
        image[:, :, 1] -= 122.95039414
        image[:, :, 2] -= 113.86538318
        image[:, :, 0] = image / 62.99321928
        image[:, :, 1] = image / 62.08870764
        image[:, :, 2] = image / 66.70489964

        # Return the image.
        return image

    def deprocess_image(self, x) -> np.ndarray:
        """
        Converts a tensor of custom-cifar10 into a valid image.

        :param x: the tensor.
        :return: the image resulting from the tensor.
        """
        # Remove extra dimension for the batches.
        x = x.reshape((self.img_nrows, self.img_ncols, 3))

        # Remove zero-center.
        x[:, :, 0] += 125.30691805
        x[:, :, 1] += 122.95039414
        x[:, :, 2] += 113.86538318
        x[:, :, 0] *= 62.99321928
        x[:, :, 1] *= 62.08870764
        x[:, :, 2] *= 66.70489964

        return x
