import os
from . import pre, csr
import imageio
from tqdm import tqdm
import numpy as np
from skimage import morphology
import pandas as pd
from .image_stats import image_summary
from skimage.feature import shape_index
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp


CPU_COUNT = int(os.environ.get('CPU_COUNT', mp.cpu_count()))


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


def process_single_image(filename, image_format, scale_metadata_path,
                         threshold_radius, smooth_radius,
                         brightness_offset, crop_radius, smooth_method):
    image = imageio.imread(filename, format=image_format)
    scale = _get_scale(image, scale_metadata_path)
    if crop_radius > 0:
        c = crop_radius
        image = image[c:-c, c:-c]
    pixel_threshold_radius = int(np.ceil(threshold_radius / scale))

    pixel_smoothing_radius = smooth_radius * pixel_threshold_radius
    thresholded = pre.threshold(image, sigma=pixel_smoothing_radius,
                                radius=pixel_threshold_radius,
                                offset=brightness_offset,
                                smooth_method=smooth_method)
    quality = shape_index(image, sigma=pixel_smoothing_radius,
                          mode='reflect')
    skeleton = morphology.skeletonize(thresholded) * quality
    framedata = csr.summarise(skeleton, spacing=scale)
    framedata['squiggle'] = np.log2(framedata['branch-distance'] /
                                    framedata['euclidean-distance'])
    framedata['scale'] = scale
    framedata.rename(columns={'mean pixel value': 'mean shape index'},
                     inplace=True)
    framedata['filename'] = filename
    return image, thresholded, skeleton, framedata


def process_images(filenames, image_format, threshold_radius,
                   smooth_radius, brightness_offset, scale_metadata_path,
                   crop_radius=0, smooth_method='Gaussian',
                   num_threads=CPU_COUNT):
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
    crop_radius : int, optional
        Crop `crop_radius` pixels from each margin of the image before
        processing.
    smooth_method : {'Gaussian', 'TV', 'NL'}, optional
        Which method to use for smoothing.
    num_threads : int, optional
        How many threads to use for computation. This should generally be
        set to the number of CPU cores available to you.

    Returns
    -------
    results : generator
        The pipeline yields individual image results in the form of a tuple
        of ``(filename, image, thresholded_image, skeleton, data_frame)``.
        Finally, after all the images have been processed, the pipeline yields
        a DataFrame containing all the collated branch-level results.
    """
    image_format = None if image_format == 'auto' else image_format
    results = []
    image_results = []
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        future_data = {ex.submit(process_single_image, filename,
                                 image_format, scale_metadata_path,
                                 threshold_radius, smooth_radius,
                                 brightness_offset, crop_radius,
                                 smooth_method): filename
                       for filename in filenames}
        for completed_data in tqdm(as_completed(future_data)):
            image, thresholded, skeleton, framedata = completed_data.result()
            filename = future_data[completed_data]
            results.append(framedata)
            image_stats = image_summary(skeleton,
                                        spacing=framedata['scale'][0])
            image_stats['filename'] = filename
            image_stats['branch density'] = (framedata.shape[0] /
                                             image_stats['area'])
            j2j = framedata[framedata['branch-type'] == 2]
            image_stats['mean J2J branch distance'] = (
                                            j2j['branch-distance'].mean())
            image_results.append(image_stats)
            yield filename, image, thresholded, skeleton, framedata
    yield pd.concat(results), pd.concat(image_results)
