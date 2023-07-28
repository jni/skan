---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Pruning Skeletons

When tracing biological molecules such as DNA using [Atomic Force
Microscopy](https://en.wikipedia.org/wiki/Atomic_force_microscopy) skeletons often have a side-branches that need
removing to leave the skeleton that represents the backbone of the molecule. Images indicate the height of points in the
x/y plane but are never as clean as one might like. The number of extraneous branches can be reduced by first applying a
Gaussian blur prior to skeletonisation.


```{code-cell} ipython3
%matplotlib inline
%config InlineBackend.figure_format='retina'

import matplotlib.pyplot as plt
import numpy as np
```

A sample image showing a loop of DNA is shown below after filtering which involves flattening and removing the inherent
gradient that is present in scans.

```{code-cell} ipython3
raw = np.load("../example-data/sample_grain.npy")
plt.imshow(raw)
```

The mask for this image is rather messy

```{code-cell} ipython3
raw_mask = np.load("../example-data/sample_grain_mask.npy")
plt.imshow(raw_mask)
```

If it is skeletonised as is then we have a skeleton with a large number of side branches and internal loops.

```{code-cell} ipython3
from skimage import morphology
raw_skeleton = morphology.skeletonize(raw_mask, method="zhang")
plt.imshow(raw_skeleton)
```

We want to remove all the side-branches which are paths that go from a junction to end-point and we can use the
`.prune_paths()` method to remove such branches, to do this we need to use the `summarize()` function which converts the
skeleton to a graph and classifies each path into one of the following types, as well calculating a number of metrics on
each path.

| Branch Type | Description                            |
|-------------|----------------------------------------|
| `0`         | endpoint-to-endpoint (isolated branch) |
| `1`         | junction-to-endpoint                   |
| `2`         | junction-to-junction                   |
| `3`         | isolated cycle                         |

```{code-cell} ipython3
import skan
from skan import Skeleton

# Summarise the skeleton
skeleton = Skeleton(raw_skeleton, keep_images=True)
skeleton_summary = skan.summarize(skeleton)
# Extract the indices of paths of type
junction_to_endpoint = skeleton_summary[skeleton_summary["branch-type"] == 1].index
skeleton_pruned = skeleton.prune_paths(junction_to_endpoint).skeleton_image
plt.imshow(skeleton_pruned)
```

However because some of the branches have branches themselves which were not removed since they dodn't terminate in an
end-point. Further we observe some small loops that also need removing.

## Iteratively Prune Paths

To address this issue the `iteratively_prune_paths()` function can be used to repeatedly prune skeletons until only a
single path remains, whether that is circular or linear.

```{code-cell} ipython3
from skan import iteratively_prune_paths

skeleton_pruned_iteratively = iteratively_prune_paths(skeleton)
plt.imshow(skeleton_pruned_iteratively.skeleton_image)
```

This has worked fairly well, it has removed all the side branches and small loops but there are a number of internal
loops that remain. This is because we started with a noisy/messy mask to derive the skeleton, but we can improve on
that.

## Gaussian Blurring

If we apply a [Gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) to the raw image before skeletonising we
obtain a much clearer mask.


```{code-cell} ipython3
from skimage import filters

raw_smoothed = filters.gaussian(raw, sigma=3)
plt.imshow(raw_smoothed)
```

To get a binary mask we use [Otsu Thresholding](https://en.wikipedia.org/wiki/Otsu%27s_method) to find a threshold above
which points are considered part of the image and below which they are not.

```{code-cell} ipython3
mask_smoothed = raw_smoothed >= filters.threshold_otsu(raw_smoothed)
plt.imshow(mask_smoothed)
```

This already looks a lot cleaner, when we skeletonise the image we now have far less side-branches and loops.

```{code-cell} ipython3
skeleton_smoothed = morphology.skeletonize(mask_smoothed, method="zhang")
plt.imshow(skeleton_smoothed)
```

In this instance the Gaussian blurring has been so effective we only have a single branch to prune and there are no
internal loops, but we will use the `iteratively_prune_paths()` function to remove these.
iterative approach to tidy this up

```{code-cell} ipython3
skeleton_smoothed_pruned_iteratively = iteratively_prune_paths(skeleton).skeleton_image
plt.imshow(skeleton_smoothed_pruned_iteratively)
```

We now have a single path for the skeleton which represents the loop of DNA.
