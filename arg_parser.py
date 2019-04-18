from argparse import ArgumentParser

# Default arguments.
ITERATIONS = 1000
CREATE_GIF = False
FRAMES = 10
DURATION = 3000
CONTENT_WEIGHT = 1.2
STYLE_WEIGHT = 1
TV_WEIGHT = 1


def create_parser() -> ArgumentParser:
    """
    Creates an argument parser for the main script.

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
    parser.add_argument('-cw', '--content_weight', type=float, default=CONTENT_WEIGHT, required=False,
                        help='Content weight (default %(default)s).')
    parser.add_argument('-sw', '--style_weight', type=float, default=STYLE_WEIGHT, required=False,
                        help='Style weight (default %(default)s).')
    parser.add_argument('-tvw', '--tv_weight', type=float, default=TV_WEIGHT, required=False,
                        help='Total Variation weight (default %(default)s).')

    return parser
