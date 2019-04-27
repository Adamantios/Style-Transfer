import os
import pickle
from typing import Union

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.optimizers import rmsprop, adam, adamax, adadelta, adagrad, sgd
from tensorflow.python.keras.utils import to_categorical

from utils import create_training_parser, create_path
from core.networks.custom_network import StyleTransferCustom


def preprocess_images(images_array):
    """
    Preprocess the given images.

    :param images_array: array containing the images to be preprocessed.
    :return: the images array preprocessed.
    """
    for i, image in enumerate(images_array):
        images_array[i, :, :, :] = StyleTransferCustom.preprocess_image(image.astype('float32'))

    return images_array


def create_model() -> Sequential:
    if start_point != '':
        if os.path.isfile(start_point):
            return StyleTransferCustom.network(input_shape=x_train.shape[1:], weights_path=start_point)
        else:
            raise FileNotFoundError('Checkpoint file \'{}\' not found.'.format(start_point))
    else:
        return StyleTransferCustom.network(input_shape=x_train.shape[1:])


def initialize_optimizer() -> Union[adam, rmsprop, sgd, adagrad, adadelta, adamax]:
    """
    Initializes an optimizer based on the user's choices.

    :return: the optimizer.
    """
    if optimizer_name == 'adam':
        return adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, decay=decay)
    elif optimizer_name == 'rmsprop':
        return rmsprop(lr=learning_rate, rho=rho, decay=decay)
    elif optimizer_name == 'sgd':
        return sgd(lr=learning_rate, momentum=momentum, decay=decay)
    elif optimizer_name == 'adagrad':
        return adagrad(lr=learning_rate, decay=decay)
    elif optimizer_name == 'adadelta':
        return adadelta(lr=learning_rate, rho=rho, decay=decay)
    elif optimizer_name == 'adamax':
        return adamax(lr=learning_rate, beta_1=beta1, beta_2=beta2, decay=decay)
    else:
        raise ValueError('An unexpected optimizer name has been encountered.')


def init_callbacks() -> []:
    callbacks = []
    if not omit_checkpoint:
        # Create path for the file.
        create_path(checkpoint_filepath)

        # Create checkpoint.
        checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=verbosity,
                                     save_best_only=True, mode='max')
        callbacks.append(checkpoint)

    return callbacks


def save_results() -> None:
    # Save weights.
    if not omit_weights:
        # Create path for the file.
        create_path(weights_filepath)
        # Save weights.
        model.save_weights(weights_filepath)
        print('Network\'s weights have been saved as {}.\n'.format(weights_filepath))

    # Save history.
    if not omit_history:
        # Create path for the file.
        create_path(weights_filepath)
        # Save history.
        with open(hist_filepath, 'wb') as file:
            pickle.dump(history.history, file)
        print('Network\'s history has been saved as {}.\n'.format(hist_filepath))


if __name__ == '__main__':
    # Get arguments.
    args = create_training_parser().parse_args()
    start_point = args.start_point
    omit_weights = args.omit_weights
    omit_checkpoint = args.omit_checkpoint
    omit_history = args.omit_history
    weights_filepath = args.weights_filepath
    hist_filepath = args.history_filepath
    checkpoint_filepath = args.checkpoint_filepath
    optimizer_name = args.optimizer
    learning_rate = args.learning_rate
    clip_norm = args.clip_norm
    clip_value = args.clip_value
    beta1 = args.beta1
    beta2 = args.beta2
    rho = args.rho
    momentum = args.momentum
    decay = args.decay
    batch_size = args.batch_size
    evaluation_batch_size = args.evaluation_batch_size
    epochs = args.epochs
    verbosity = args.verbosity

    # Load dataset.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocess data.
    x_train = preprocess_images(x_train.copy())
    x_test = preprocess_images(x_test.copy())
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create model.
    model = create_model()

    # Initialize optimizer.
    optimizer = initialize_optimizer()

    # Compile model.
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Initialize callbacks list.
    callbacks_list = init_callbacks()

    # Fit network.
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list,
                        validation_data=(x_test, y_test))

    # Save results.
    save_results()
