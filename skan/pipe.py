from . import pre, csr
import imageio
import tqdm
import numpy as np
from skimage import morphology
import pandas as pd


def process_images(filenames, image_format, threshold_radius,
                   smooth_radius, brightness_offset, scale_metadata_path):
    image_format = None if image_format == 'auto' else image_format
    results = []
    for file in tqdm(filenames):
        image = imageio.imread(file, format=image_format)
        if scale_metadata_path is not None:
            md_path = scale_metadata_path.split(sep=',')
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
