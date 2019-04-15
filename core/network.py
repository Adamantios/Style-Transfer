import numpy as np
from tensorflow.contrib.keras.api.keras.preprocessing.image import img_to_array
from tensorflow.contrib.keras.api.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.contrib.keras.api.keras import backend as K
from PIL.Image import Image


class StyleTransferVGG19(object):

    def __init__(self, content_image: Image, style_image: Image):
        # Get the content image's width and height.
        width, height = content_image.size

        # Resize the style image corresponding to the content image.
        style_image = style_image.resize((width, height))

        # Se the dimensions of the generated picture.
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

        # Build the VGG19 network, using pre-trained ImageNet weights.
        model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        self._outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

        self.content_features_layer = self._outputs_dict['block5_conv2']
        self.style_features_layers = [self._outputs_dict[layer_name] for layer_name in ['block1_conv1', 'block2_conv1',
                                                                                        'block3_conv1', 'block4_conv1',
                                                                                        'block5_conv1']]

    # util function to open, resize and format pictures into appropriate tensors

    @staticmethod
    def preprocess_image(image):
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        # image = tf.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return image

    # util function to convert a tensor into a valid image

    def deprocess_image(self, x) -> np.ndarray:
        if K.image_data_format() == 'channels_first':
            x = x.reshape((3, self.img_nrows, self.img_ncols))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((self.img_nrows, self.img_ncols, 3))
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x
