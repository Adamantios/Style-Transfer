from os import path, makedirs
from argparse import ArgumentParser

# ----------------------------------- DEFAULT ARGUMENTS ------------------------------------------

# ------------------------------------- STYLE TRANSFER -------------------------------------------
ITERATIONS = 1000
CREATE_GIF = False
FRAMES = 10
DURATION = 3000
LOOP = 0
CONTENT_WEIGHT = 50
STYLE_WEIGHT = 100
TV_WEIGHT = 100
WEIGHTS_PATH = ''
NETWORK = 'vgg'
NETWORK_CHOICES = 'vgg', 'custom'

# ---------------------------------------- TRAINING ----------------------------------------------
START_POINT = ''
SAVE_WEIGHTS = True
SAVE_CHECKPOINT = True
SAVE_HIST = True
WEIGHTS_FILENAME = 'out/network_weights.h5'
CHECKPOINT_FILENAME = 'out/checkpoint.h5'
HIST_FILENAME = 'out/train_history.pickle'
OPTIMIZER = 'rmsprop'
OPTIMIZER_CHOICES = 'adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax'
LEARNING_RATE = 1E-3
CLIP_NORM = 1
CLIP_VALUE = .5
BETA1 = .9
BETA2 = .999
RHO = .9
MOMENTUM = .0
DECAY = 1E-6
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
EPOCHS = 125
VERBOSITY = 1


# ------------------------------------------------------------------------------------------------

def create_style_transfer_parser() -> ArgumentParser:
    """
    Creates an argument parser for the style transfer script.

    :return: ArgumentParser object.
    """
    parser = ArgumentParser(description='Style transfer using VGG19 feature layers and L-BFGS optimizer.',
                            epilog='Note: '
                                   'The result image is going to have the same dimensions with the content image.')
    parser.add_argument('content_image_path', metavar='content', type=str,
                        help='Path to the content image.')
    parser.add_argument('style_image_path', metavar='style', type=str,
                        help='Path to the style image.')
    parser.add_argument('combined_filename', metavar='result', type=str,
                        help='Prefix for the saved image results. '
                             'Please omit the extension, which is chosen automatically (.png). '
                             'Creates the path to the file if it does not exist.')
    parser.add_argument('-i', '--iter', type=int, default=ITERATIONS, required=False,
                        help='Number of iterations for the optimizer (default %(default)s).\n'
                             'If --gif is passed, this is the number of iterations for each frame.')
    parser.add_argument('-g', '--gif', default=CREATE_GIF, required=False, action='store_true',
                        help='Whether a gif of the procedure should be created (default %(default)s).')
    parser.add_argument('-f', '--frames', type=int, default=FRAMES, required=False,
                        help='Number of the gif frames (default %(default)s).\nIgnored if --gif is not passed.')
    parser.add_argument('-d', '--duration', type=int, default=DURATION, required=False,
                        help='Duration of the gif in milliseconds (default %(default)s).\n'
                             'Ignored if --gif is not passed.')
    parser.add_argument('-l', '--loop', default=LOOP, required=False,
                        help='Number of times the gif should loop (default %(default)s).\n'
                             'For infinite loop, the number 0 should be passed.')
    parser.add_argument('-cw', '--content_weight', type=float, default=CONTENT_WEIGHT, required=False,
                        help='Content weight (default %(default)s).')
    parser.add_argument('-sw', '--style_weight', type=float, default=STYLE_WEIGHT, required=False,
                        help='Style weight (default %(default)s).')
    parser.add_argument('-tvw', '--tv_weight', type=float, default=TV_WEIGHT, required=False,
                        help='Total Variation weight (default %(default)s).')
    parser.add_argument('-n', '--network', type=str.lower, default=NETWORK, required=False, choices=NETWORK_CHOICES,
                        help='The network to be used. (default %(default)s).')
    parser.add_argument('-p', '--path', type=str, default=WEIGHTS_PATH, required=False,
                        help='The network\'s weights path. (default %(default)s).\n'
                             'Use this parameter, if you have the network\'s weights saved locally, '
                             'so that you do not have to wait for the model to download.\n'
                             'It should be the wights of the network that was specified with the '
                             '\'--network\' parameter. \n'
                             'WARNING: If you choose the custom network, the weights have to be passed.'
                             'If you choose the VGG network, this parameter may be ignored '
                             'and the weights will be automatically downloaded.')

    return parser


def create_training_parser() -> ArgumentParser:
    """
    Creates an argument parser for the network training script.

    :return: ArgumentParser object.
    """
    parser = ArgumentParser(description='Training the custom CNN network, to be used for style transferring.',
                            epilog='Note: '
                                   'The hyperparameters will be ignored if the chosen optimizer does not use them.')
    parser.add_argument('-sp', '--start_point', type=str, required=False, default=START_POINT,
                        help='Filepath containing existing weights to initialize the model.')
    parser.add_argument('-ow', '--omit_weights', default=not SAVE_WEIGHTS, required=False, action='store_false',
                        help='Whether the weights should not be saved (default %(default)s).')
    parser.add_argument('-oc', '--omit_checkpoint', default=not SAVE_CHECKPOINT, required=False, action='store_false',
                        help='Whether the best weights checkpoint should not be saved (default %(default)s).')
    parser.add_argument('-oh', '--omit_history', default=not SAVE_HIST, required=False, action='store_false',
                        help='Whether the training history should not be saved (default %(default)s).')
    parser.add_argument('-wf', '--weights_filepath', default=WEIGHTS_FILENAME, required=False, type=str,
                        help='Path to store the trained network\'s weights (default %(default)s). '
                             'Ignored if --omit_weights has been chosen')
    parser.add_argument('-hf', '--history_filepath', default=HIST_FILENAME, required=False, type=str,
                        help='Path to store the trained network\'s history (default %(default)s). '
                             'Ignored if --omit_history has been chosen')
    parser.add_argument('-cf', '--checkpoint_filepath', default=CHECKPOINT_FILENAME, required=False, type=str,
                        help='Path to store the trained network\'s best checkpoint(default %(default)s). '
                             'Ignored if --omit_checkpoint has been chosen')
    parser.add_argument('-o', '--optimizer', type=str.lower, default=OPTIMIZER, required=False,
                        choices=OPTIMIZER_CHOICES,
                        help='The optimizer to be used. (default %(default)s).')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, required=False,
                        help='The learning rate for the optimizer (default %(default)s).\n')
    parser.add_argument('-cn', '--clip_norm', type=float, default=CLIP_NORM, required=False,
                        help='The clip norm for the optimizer (default %(default)s).\n')
    parser.add_argument('-cv', '--clip_value', type=float, default=CLIP_VALUE, required=False,
                        help='The clip value for the optimizer (default %(default)s).\n')
    parser.add_argument('-b1', '--beta1', type=float, default=BETA1, required=False,
                        help='The beta 1 for the optimizer (default %(default)s).\n')
    parser.add_argument('-b2', '--beta2', type=float, default=BETA2, required=False,
                        help='The beta 2 for the optimizer (default %(default)s).\n')
    parser.add_argument('-rho', type=int, default=float, required=False,
                        help='The rho for the optimizer (default %(default)s).\n')
    parser.add_argument('-m', '--momentum', type=float, default=MOMENTUM, required=False,
                        help='The momentum for the optimizer (default %(default)s).\n')
    parser.add_argument('-d', '--decay', type=float, default=DECAY, required=False,
                        help='The decay for the optimizer (default %(default)s).\n')
    parser.add_argument('-bs', '--batch_size', type=int, default=TRAIN_BATCH_SIZE, required=False,
                        help='The batch size for the optimization (default %(default)s).\n')
    parser.add_argument('-ebs', '--evaluation_batch_size', type=int, default=EVAL_BATCH_SIZE, required=False,
                        help='The batch size for the evaluation (default %(default)s).\n')
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS, required=False,
                        help='The number of epochs to train the network (default %(default)s).\n')
    parser.add_argument('-v', '--verbosity', type=int, default=VERBOSITY, required=False,
                        help='The verbosity for the optimization procedure (default %(default)s).\n')

    return parser


def create_path(filepath: str) -> None:
    """
    Creates a path to a file, if it does not exist.

    :param filepath: the filepath.
    """
    # Get the file's directory.
    directory = path.dirname(filepath)

    # Create directory if it does not exist
    if not path.exists(directory):
        makedirs(directory)
