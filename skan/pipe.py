import os
from . import pre, csr
import imageio
from tqdm import tqdm
import numpy as np
from skimage import morphology
import pandas as pd
from . import draw
from .image_stats import image_summary
import matplotlib.pyplot as plt


def _get_scale(image, md_path_or_scale):
    """Get a valid scale from an image and a metadata path or scale."""
    scale = None
    try:
        scale = float(md_path_or_scale)
    except ValueError:
        pass
    if md_path_or_scale is not None and scale is None:
        md_path = md_path_or_scale.split(sep='/')
        meta = image.meta
        for key in md_path:
            meta = meta[key]
        scale = float(meta)
    else:
        if scale is None:
            scale = 1  # measurements will be in pixel units
    return scale


def process_images(filenames, image_format, threshold_radius,
                   smooth_radius, brightness_offset, scale_metadata_path,
                   save_skeleton='', output_folder=None):
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
    save_skeleton : string, optional
        If this is not an empty string, skeleton plots will be saved with
        the given prefix, one per input filename.

    Returns
    -------
    result : pandas DataFrame
        Data frame containing all computed statistics on the skeletons found
        in the input image.
    """
    image_format = None if image_format == 'auto' else image_format
    results = []
    image_results = []
    for file in tqdm(filenames):
        image = imageio.imread(file, format=image_format)
        scale = _get_scale(image, scale_metadata_path)
        pixel_threshold_radius = int(np.ceil(threshold_radius / scale))
        pixel_smoothing_radius = smooth_radius * pixel_threshold_radius
        thresholded = pre.threshold(image, sigma=pixel_smoothing_radius,
                                    radius=pixel_threshold_radius,
                                    offset=brightness_offset)
        skeleton = morphology.skeletonize(thresholded)
        framedata = csr.summarise(skeleton, spacing=scale)
        framedata['squiggle'] = np.log2(framedata['branch-distance'] /
                                        framedata['euclidean-distance'])
        framedata['filename'] = file
        framedata['scale'] = scale
        results.append(framedata)
        if save_skeleton:
            fig, axes = draw.pipeline_plot(image, sigma=pixel_smoothing_radius,
                                           radius=pixel_threshold_radius,
                                           offset=brightness_offset)
            output_basename = (save_skeleton +
                               os.path.basename(os.path.splitext(file)[0]) +
                               '.png')
            output_filename = os.path.join(output_folder, output_basename)
            fig.savefig(output_filename, dpi=300)
            plt.close(fig)
        image_stats = image_summary(skeleton, spacing=scale)
        image_stats['filename'] = file
        image_stats['branch density'] = (framedata.shape[0] /
                                         image_stats['area'])
        image_results.append(image_stats)

    return pd.concat(results), pd.concat(image_results)
