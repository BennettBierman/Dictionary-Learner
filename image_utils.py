import patchify
from skimage import color
import matplotlib.pyplot as plt
"""
Utility functions for working with image data
"""


def normalize(image):
    """
    Transforms color images to black and white and normalizes all
    images to ensure a wide range of pixels
    :param image: d
    :return: normalized image
    """
    if len(image.shape) > 2:
        img = color.rgb2gray(image)
        return (img - img.min()) / (img.max() - img.min())
    return (image - image.min()) / (image.max() - image.min())


def patch_img(image, patch_size):
    """
    Creates a representation of an image where each column is a patch
    of the original image
    :param image
    :param patch_size: width and height of each path
    :return: image representation
    """
    patches = patchify.patchify(image, (patch_size, patch_size), step=1)
    r, c, x, y = patches.shape
    img_rep = patches.reshape((x*y, r*c))
    return img_rep, (r, c, x, y)


def unpatch_img(img_rep, dim):
    """
    Reconstructs original image from representation produced with
    the patch_img function
    :param img_rep: image representation created by patch_img
    :param dim: shape of the patchified version of the original image
    :return: data in original image form
    """
    (r, c, x, y) = dim
    patches = img_rep.reshape((r, c, x, y))
    image = patchify.unpatchify(patches, (r+x-1, c+y-1))
    return image


def show_images(images, size):
    """
    Plots a list of images on the same axis
    :param images: list of images
    :param size: size of each image plot
    """
    fig, axis = plt.subplots(figsize=(size, size), ncols=len(images))
    if len(images) >= 2:
        for i in range(len(images)):
            img = images[i]
            axis[i].imshow(img, cmap="gray")
    else:
        axis.imshow(images[0], cmap='gray')
    plt.show(block=True)
