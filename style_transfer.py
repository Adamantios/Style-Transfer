import time

from keras_preprocessing.image import load_img, save_img

from core.l_bfgs_optimizer import LBFGSOptimizer
from core.network import StyleTransferVGG19
from arg_parser import create_parser
from core.loss_calculator import LossCalculator
from PIL import Image


def style_transfer():
    optimizer = LBFGSOptimizer(content_image.height, content_image.width, f_outs)

    x = StyleTransferVGG19.preprocess_image(content_image).flatten()
    frames = []
    for i in range(args.iter):
        print('Start of iteration', i + 1)
        start_time = time.time()
        x, f = optimizer.run(x, 20)
        print('Current loss value:', f)
        # save current generated image
        img = model.deprocess_image(x)
        frames.append(Image.fromarray(img))
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i + 1, end_time - start_time))

    filename = args.combined_filename + '.png'
    save_img(filename, frames[-1])
    print("Image saved as '{}'".format(filename))

    if args.gif:
        filename = args.combined_filename + '.gif'

        try:
            # Save into a GIF file that loops forever
            frames[0].save(filename, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
            print("A GIF of the style transfer steps has been saved as '{}'".format(filename))
        except ValueError or IOError:
            print('Something went wrong while trying to save the gif.')


if __name__ == '__main__':
    # Get arguments.
    args = create_parser().parse_args()
    content_image_path = args.content_image_path
    style_image_path = args.style_image_path
    combined_filename = args.combined_filename
    gif = args.gif
    n_iter = args.iter
    content_weight = args.content_weight
    style_weight = args.style_weight
    tv_weight = args.tv_weight

    # Load images.
    # TODO check for invalid path.
    content_image = load_img(content_image_path)
    style_image = load_img(style_image_path)

    # Initialize model.
    model = StyleTransferVGG19(content_image, style_image)

    # Initialize loss calculator.
    loss_calculator = LossCalculator(model.combination_image, content_image.height, content_image.width,
                                     model.content_features_layer, model.style_features_layers,
                                     args.content_weight, args.style_weight, args.tv_weight)
    # Get loss and grads function.
    f_outs = loss_calculator.get_loss_and_grads()

    # Begin style transferring procedure.
    style_transfer()

# TODO save model and load it if it exists. Otherwise, download it.
