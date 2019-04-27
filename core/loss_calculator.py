from tensorflow import SparseTensor
import tensorflow.contrib.keras.api.keras.backend as K


class InvalidDimensionsError(Exception):
    pass


class LossCalculator(object):
    def __init__(self, combination_image: SparseTensor, img_nrows: float, img_ncols: float,
                 content_features_layer, style_features_layers: list,
                 content_weight: float, style_weight: float, total_variation_weight: float):
        # Check the combined image's dimensions.
        if K.ndim(combination_image) != 4:
            raise InvalidDimensionsError('Combination image tensor should be 4 dimensional.\n'
                                         'Got {} instead.'
                                         .format(combination_image.shape))

        self._combination_image = combination_image
        self._img_nrows = img_nrows
        self._img_ncols = img_ncols
        self._content_features_layer = content_features_layer
        self._style_features_layers = style_features_layers
        self._content_weight = content_weight
        self._style_weight = style_weight
        self._total_variation_weight = total_variation_weight

    @staticmethod
    def _gram_matrix(x):
        """
        Calculates the Gram matrix of a tensor.

        :param x: the given tensor.
        :return: the Gram matrix.
        """
        if K.ndim(x) != 3:
            raise InvalidDimensionsError('Input tensor should be 3 dimensional.\n'
                                         'Got {} instead.'
                                         .format(x.shape))

        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram

    def _style_loss(self, style, combined):
        """
        Calculates the style loss, which is the Euclidean distance
        between the gram matrices of the feature maps of the style image
        and the combined image.

        :param style: the style features.
        :param combined: the combined features.
        :return: the style loss.
        """
        if K.ndim(style) != 3:
            raise InvalidDimensionsError('Input tensor should be 3 dimensional.\n'
                                         'Got {} instead.'
                                         .format(style.shape))
        if K.ndim(combined) != 3:
            raise InvalidDimensionsError('Input tensor should be 3 dimensional.\n'
                                         'Got {} instead.'
                                         .format(combined.shape))

        s = self._gram_matrix(style)
        g = self._gram_matrix(combined)
        channels = 3
        size = self._img_nrows * self._img_ncols

        return K.sum(K.square(s - g)) / (4. * (channels ** 2) * (size ** 2))

    @staticmethod
    def _content_loss(content, combined):
        """
        Calculates the content loss, which is the Euclidean distance
        between the outputs of the model for the content image
        and the combined image.

        :param content: the feature map of the content image.
        :param combined: the feature map of the combined image.
        :return: the content loss.
        """
        return K.sum(K.square(combined - content))

    def _total_variation_loss(self):
        """
        Calculates the total variation loss.
        Keeps the generated image locally coherent.

        :return: the total variation loss.
        """
        a = K.square(
            self._combination_image[:, :self._img_nrows - 1, :self._img_ncols - 1, :] -
            self._combination_image[:, 1:, :self._img_ncols - 1, :])
        b = K.square(
            self._combination_image[:, :self._img_nrows - 1, :self._img_ncols - 1, :] -
            self._combination_image[:, :self._img_nrows - 1, 1:, :])

        return K.sum(K.pow(a + b, 1.25))

    def _total_loss(self):
        """ Calculates total loss. """
        # Initialize a loss variable as float.
        loss = 0.

        # Get the content and combination features of the content layer.
        content_features = self._content_features_layer[0, :, :, :]
        combination_features = self._content_features_layer[2, :, :, :]
        # Calculate and add the content loss.
        loss += self._content_weight * self._content_loss(content_features, combination_features)

        # Calculate the style weight per layer.
        style_weight_per_layer = self._style_weight / len(self._style_features_layers)
        for layer_features in self._style_features_layers:
            # Get the style and combination features of the content layer.
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            # Calculate and add the style loss.
            loss += style_weight_per_layer * self._style_loss(style_features, combination_features)

        # Calculate and add the total variation loss.
        loss += self._total_variation_weight * self._total_variation_loss()

        return loss

    def get_loss_and_grads(self) -> K.function:
        """
        Returns function which calculates the loss and gradient outputs.

        :return: Keras function.
        """
        loss = self._total_loss()

        # Calculate the gradients of the loss with respect to the combined image.
        grads = K.gradients(loss, self._combination_image)

        # Create outputs, containing the loss and the grads.
        outputs = [loss]
        if isinstance(grads, (list, tuple)):
            outputs += grads
        else:
            outputs.append(grads)

        # Return outputs function.
        return K.function([self._combination_image], outputs)
