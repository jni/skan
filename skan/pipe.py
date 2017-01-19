from . import pre, csr
import imageio
from tqdm import tqdm
import numpy as np
from skimage import morphology
import pandas as pd


def process_images(filenames, image_format, threshold_radius,
                   smooth_radius, brightness_offset, scale_metadata_path):
    """Full pipeline from images to skeleton stats with local median threshold.

    Parameters
    ----------
    filenames : list of string
        The list of input filenames.
    image_format : string
        The format of the files. 'auto' is automatically determined by the
        imageio library. See imageio documentation for valid image formats.
    threshold_radius : float
        The radius for median thresholding,
    smooth_radius : float in [0, 1]
        The value of sigma with which to Gaussian-smooth the image,
        **relative to `threshold_radius`**.
    brightness_offset : float
        The standard brightness value with which to threshold is the local
        median, `m(x, y)`. Use this value to offset from there: the threshold
        used will be `m(x, y) + brightness_offset`.
    scale_metadata_path : string
        The path in the image dictionary to find the metadata on pixel scale,
        separated by forward slashes ('/').

    Returns
    -------
    result : pandas DataFrame
        Data frame containing all computed statistics on the skeletons found
        in the input image.
    """
    image_format = None if image_format == 'auto' else image_format
    results = []
    for file in tqdm(filenames):
        image = imageio.imread(file, format=image_format)
        if scale_metadata_path is not None:
            md_path = scale_metadata_path.split(sep='/')
            meta = image.meta
            for key in md_path:
                meta = meta[key]
            scale = float(meta)
        else:
            scale = 1  # measurements will be in pixel units
        pixel_threshold_radius = int(np.ceil(threshold_radius / scale))
        pixel_smoothing_radius = smooth_radius * pixel_threshold_radius
        thresholded = pre.threshold(image, sigma=pixel_smoothing_radius,
                                    radius=pixel_threshold_radius,
                                    offset=brightness_offset)
        skeleton = morphology.skeletonize(thresholded)
        framedata = csr.summarise(skeleton, spacing=scale)
        framedata['squiggle'] = np.log2(framedata['branch-distance'] /
                                        framedata['euclidean-distance'])
        framedata['filename'] = [file] * len(framedata)
        results.append(framedata)
    return pd.concat(results)
