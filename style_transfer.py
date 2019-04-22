import os
import time

from keras_preprocessing.image import load_img, save_img

from core.l_bfgs_optimizer import LBFGSOptimizer
from core.vgg_network import StyleTransferVGG19
from arg_parser import create_parser
from core.loss_calculator import LossCalculator
from PIL import Image


def style_transfer():
    # Initialize optimizer.
    optimizer = LBFGSOptimizer(content_image.height, content_image.width, f_outs)

    # Preprocess image using the model's procedure and flatten it.
    x = model.preprocess_image(content_image).flatten()

    # Create image's filename.
    img_filename = combined_filename + '.png'

    if gif:
        # Initialize a frames list, containing the content image.
        frames = [content_image]
        # Create gif's filename.
        gif_filename = combined_filename + '.gif'
        for frame in range(n_frames):
            print('Creating frame {}'.format(frame + 1))
            # Start timer.
            start_time = time.time()
            # Run the optimizer.
            x, f = optimizer.run(x, n_iter)
            print('Current loss value:', f)
            # Deprocess result, in order to get a valid image. Pass a copy of x, because it is mutable!
            img = model.deprocess_image(x.copy())
            # Save current frame.
            frames.append(Image.fromarray(img))
            # Stop timer.
            end_time = time.time()
            print('Frame {} created in {} seconds'.format(frame + 1, end_time - start_time))

        try:
            # Save frames into a GIF file that loops forever.
            frames[0].save(gif_filename, format='GIF', append_images=frames[1:], save_all=True,
                           duration=int(gif_duration / len(frames)), loop=loop)
            print("A GIF of the style transfer steps has been saved as '{}'".format(gif_filename))
        except ValueError or IOError:
            print('Something went wrong while trying to save the gif.')

        # Save result.
        save_img(img_filename, frames[-1])
        print("Result image saved as '{}'".format(img_filename))

    else:
        print('Starting optimisation.')
        # Start timer.
        start_time = time.time()
        # Run the optimizer.
        x, f = optimizer.run(x, n_iter)
        print('Current loss value:', f)
        # Stop timer.
        end_time = time.time()
        print('Optimisation finished in {} seconds'.format(end_time - start_time))
        # Deprocess result, in order to get a valid image.
        img = model.deprocess_image(x)
        # Save the image.
        save_img(img_filename, img)
        print("Image saved as '{}'".format(img_filename))


if __name__ == '__main__':
    # Get arguments.
    args = create_parser().parse_args()
    content_image_path = args.content_image_path
    style_image_path = args.style_image_path
    combined_filename = args.combined_filename
    n_iter = args.iter
    gif = args.gif
    n_frames = args.frames
    gif_duration = args.duration
    loop = args.loop
    content_weight = args.content_weight
    style_weight = args.style_weight
    tv_weight = args.tv_weight
    network = args.network
    path = args.path

    if not os.path.isfile(path):
        raise FileNotFoundError('Network weights file {} does not exist.'.format(path))

    # Load images.
    content_image = load_img(content_image_path)
    style_image = load_img(style_image_path)

    # Initialize model.
    if network == 'vgg':
        model = StyleTransferVGG19(content_image, style_image, path)
    elif network == 'custom':
        model = StyleTransferVGG19(content_image, style_image, path)
        # TODO model = StyleTransferCustom(content_image, style_image, path)
    else:
        raise ValueError('Invalid parameter has been encountered for the \'--network\' argument.')

    # Initialize loss calculator.
    loss_calculator = LossCalculator(model.combination_image, content_image.height, content_image.width,
                                     model.content_features_layer, model.style_features_layers,
                                     content_weight, style_weight, tv_weight)
    # Get loss and grads function.
    f_outs = loss_calculator.get_loss_and_grads()

    # Begin style transferring procedure.
    style_transfer()
