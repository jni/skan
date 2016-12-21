from . import pre, csr
import imageio
import tqdm
import numpy as np
from skimage import morphology
import pandas as pd


def process_images(filenames, image_format, threshold_radius,
                   smooth_radius, brightness_offset, scale_metadata_path):
    image_format = (None if self.image_format.get() == 'auto'
                    else self.image_format.get())
    results = []
    from skan import pre, csr
    for file in tqdm(filenames):
        image = imageio.imread(file, format=image_format)
        if self.scale_metadata_path is not None:
            md_path = self.scale_metadata_path.get().split(sep=',')
            meta = image.meta
            for key in md_path:
                meta = meta[key]
            scale = float(meta)
        else:
            scale = 1  # measurements will be in pixel units
        pixel_threshold_radius = int(np.ceil(self.threshold_radius.get() /
                                             scale))
        pixel_smoothing_radius = (self.smooth_radius.get() *
                                  pixel_threshold_radius)
        thresholded = pre.threshold(image, sigma=pixel_smoothing_radius,
                                    radius=pixel_threshold_radius,
                                    offset=self.brightness_offset.get())
        skeleton = morphology.skeletonize(thresholded)
        framedata = csr.summarise(skeleton, spacing=scale)
        framedata['squiggle'] = np.log2(framedata['branch-distance'] /
                                        framedata['euclidean-distance'])
        framedata['filename'] = [file] * len(framedata)
        results.append(framedata)
    results = pd.concat(results)
