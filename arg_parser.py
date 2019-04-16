from argparse import ArgumentParser


def create_parser() -> ArgumentParser:
    """
    Creates an argument parser for the main script.

    :return: ArgumentParser object.
    """
    parser = ArgumentParser(description='Style transfer using VGG19 feature layers and L-BFGS optimizer.')
    parser.add_argument('content_image_path', metavar='content', type=str,
                        help='Path to the content image.')
    parser.add_argument('style_image_path', metavar='style', type=str,
                        help='Path to the style image.')
    parser.add_argument('combined_filename', metavar='result', type=str,
                        help='Prefix for the saved image results.')
    parser.add_argument('-i', '--iter', type=int, default=20, required=False,
                        help='Number of iterations for the optimizer.\n'
                             'If --gif is passed, this is the number of iterations for each frame.')
    parser.add_argument('-g', '--gif', default=False, required=False, action='store_true',
                        help='Whether a gif of the procedure should be created.')
    parser.add_argument('-f', '--frames', type=int, default=10, required=False,
                        help='Number of the gif frames.\nIgnored if --gif is not passed.')
    parser.add_argument('-cw', '--content_weight', type=float, default=1.2, required=False,
                        help='Content weight.')
    parser.add_argument('-sw', '--style_weight', type=float, default=1.0, required=False,
                        help='Style weight.')
    parser.add_argument('-tvw', '--tv_weight', type=float, default=1.0, required=False,
                        help='Total Variation weight.')

    return parser
