import skimage
import numpy as np

from typing import Any, NewType, Union

# Define custom types for image formats
# This helps us avoid bugs where the wrong format is passed around
ScikitImageArray = NewType("ScikitImageArray", np.ndarray)
TorchImageArray = NewType("TorchImageArray", np.ndarray)
AnyImageArray = Union[ScikitImageArray, TorchImageArray]

class ImageFormat:
    """
    A converter between image formats.

    Definitions of formats:
    `scikit`: Channels last, int8, [0, 255]
    `torch`: Channels first, float32, [-1, 1]
    """

    @staticmethod
    def scikit(imgs: AnyImageArray) -> ScikitImageArray:
        """
        Convert image(s) to scikit format. Autodetects current image format
        """
        fmt = ImageFormat.detect_format(imgs)
        if fmt == 'scikit':
            return imgs
        elif fmt == 'torch':
            return ImageFormat.torch2scikit(imgs)
    
    @staticmethod
    def torch(imgs: AnyImageArray) -> TorchImageArray:
        """
        Convert image(s) to torch format. Autodetects current image format
        """
        fmt = ImageFormat.detect_format(imgs)
        if fmt == 'scikit':
            return ImageFormat.scikit2torch(imgs)
        elif fmt == 'torch':
            return imgs 


    @staticmethod
    def detect_format(imgs: AnyImageArray) -> str:
        # Using -3/-1 instead of 0/3 means that multiple images can be processed or just one
        if 1 <= imgs.shape[-1] <= 4:
            return 'scikit'
        elif 1 <= imgs.shape[-3] <= 4:
            return 'torch'
        else:
            raise ValueError('Provided image format could not be detected')

    @staticmethod
    def torch2scikit(imgs: TorchImageArray) -> ScikitImageArray:
        out = np.moveaxis(imgs.copy(), -3, -1)
        out += 1
        out *= 127.5
        return np.rint(out).astype(np.uint8)

    @staticmethod
    def scikit2torch(imgs: ScikitImageArray) -> TorchImageArray:
        out = np.moveaxis(imgs.copy(), -1, -3).astype(np.float32)
        out /= 127.5
        out -= 1
        return out

def overlay_transparent_patch(img1: AnyImageArray, img2: AnyImageArray) -> ScikitImageArray:
    """
    Overlays a transparent patch on top of a 
    
    The second image `img2` must have an alpha channel.
    The first image `img1` may or may not have an alpha channel.
    
    Both input images MUST be in the scikit format.
    The resulting image will be returned in scikit format (alpha channel in the range 0-255, RGBA, channels last)
    """
    img1 = ImageFormat.scikit(img1)
    img2 = ImageFormat.scikit(img2)

    assert img1.shape[:2] == img2.shape[:2], f"Images must be the same dimensions, got {img1.shape} {img2.shape}"
    alpha = (img2[:, :, 3]).astype(float) / 255
    print(alpha)
    inv_alpha = 1. - alpha

    img_out = np.zeros_like(img1)

    # Binary mask each channel in turn
    for channel in range(3):
        img_out[:, :, channel] = (inv_alpha) * img1[:, :, channel] + (alpha) * img2[:, :, channel]

    # Preserve alpha channel
    if img1.shape[2] == 4:
        img_out[:, :, 3] = np.maximum(img1[:, :, 3], img2[:, :, 3])

    return img_out

    
    