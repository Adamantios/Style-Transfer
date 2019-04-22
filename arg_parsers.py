from argparse import ArgumentParser

# Default arguments.
ITERATIONS = 1000
CREATE_GIF = False
FRAMES = 10
DURATION = 3000
LOOP = 0
CONTENT_WEIGHT = 1.2
STYLE_WEIGHT = 1
TV_WEIGHT = 1
WEIGHTS_PATH = ''
NETWORK = 'vgg'
NETWORK_CHOICES = 'vgg', 'custom'


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
                        help='Prefix for the saved image results.')
    parser.add_argument('-i', '--iter', type=int, default=ITERATIONS, required=False,
                        help='Number of iterations for the optimizer (default %(default)s).\n'
                             'If --gif is passed, this is the number of iterations for each frame.')
    parser.add_argument('-g', '--gif', default=CREATE_GIF, required=False, action='store_true',
                        help='Whether a gif of the procedure should be created.')
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
    parser.add_argument('-n', '--network', type=str, default=NETWORK, required=False, choices=NETWORK_CHOICES,
                        help='The network to be used. (default %(default)s).')
    parser.add_argument('-p', '--path', type=str, default=WEIGHTS_PATH, required=False,
                        help='The network\'s weights path. (default %(default)s).\n'
                             'Use this parameter, if you have the network\'s weights saved locally, '
                             'so that you do not have to wait for the model to download.\n'
                             'It should be the wights of the network that was specified with the '
                             '\'--network\' parameter.')

    return parser


def create_training_parser() -> ArgumentParser:
    """
    Creates an argument parser for the network training script.

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
                        help='Prefix for the saved image results.')
    parser.add_argument('-i', '--iter', type=int, default=ITERATIONS, required=False,
                        help='Number of iterations for the optimizer (default %(default)s).\n'
                             'If --gif is passed, this is the number of iterations for each frame.')
    parser.add_argument('-g', '--gif', default=CREATE_GIF, required=False, action='store_true',
                        help='Whether a gif of the procedure should be created.')
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
    parser.add_argument('-n', '--network', type=str, default=NETWORK, required=False, choices=NETWORK_CHOICES,
                        help='The network to be used. (default %(default)s).')
    parser.add_argument('-p', '--path', type=str, default=WEIGHTS_PATH, required=False,
                        help='The network\'s weights path. (default %(default)s).\n'
                             'Use this parameter, if you have the network\'s weights saved locally, '
                             'so that you do not have to wait for the model to download.\n'
                             'It should be the wights of the network that was specified with the '
                             '\'--network\' parameter.')

    return parser