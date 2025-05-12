---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
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

```{code-cell} ipython3
np.max(raw)
```

The mask for this image is rather messy

```{code-cell} ipython3
from skimage import filters

raw_mask = filters.threshold_li(raw) < raw
plt.imshow(raw_mask)
```

Keeping only the largest component, skeletonizing as-is will give a skeleton with a large number of side branches and internal loops. (We multiply the mask by the raw image to get the height of each point, since this is an AFM image.)

```{code-cell} ipython3
from skimage import morphology
large_comp = morphology.remove_small_objects(raw_mask, min_size=100)
raw_skeleton = morphology.skeletonize(large_comp, method="zhang")
plt.imshow(raw_skeleton * raw)
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

height_skeleton = raw_skeleton * raw
# Summarise the skeleton
skeleton = Skeleton(height_skeleton, keep_images=True, value_is_height=True)  # TODO: include pixel spacing
skeleton_summary = skan.summarize(skeleton, separator='-')
# Extract the indices of paths of type
junction_to_endpoint = skeleton_summary[skeleton_summary["branch-type"] == 1].index
skeleton_pruned = skeleton.prune_paths(junction_to_endpoint).skeleton_image
plt.imshow(skeleton_pruned)
```

However because some of the branches have branches themselves which were not removed since they dodn't terminate in an
end-point. Further we observe some small loops that also need removing.

## Iteratively Prune Paths

To address this issue the `iteratively_prune_paths()` function can be used to repeatedly prune skeletons until only a
single path remains, whether that is circular or linear. However, this function needs the skeleton in another format,
a networkx MultiGraph, because the native data structures in skan are not easy to update repeatedly. We use the `skeleton_to_nx`
function for this.

```{code-cell} ipython3
from skan.csr import skeleton_to_nx  # TODO: update to just skan
```

```{code-cell} ipython3
nxskel = skeleton_to_nx(skeleton, skeleton_summary)
```

Now, we can use the iteratively_prune_paths function. This function takes in the graph, and a *discard* predicate, a function that takes as input the graph and an edge ID, and returns True if that edge should be removed. To create a useful predicate, it's good to know what attributes an edge has, which we can do by inspecting an arbitrary edge in our graph. Note the keywords `data=True`, which returns the full data on each edge, and `keys=True`, which returns, in addition to the edge source and target, a *key* for each edge, which distinguishes edges when there are multiple edges between two nodes.

```{code-cell} ipython3
next(iter(nxskel.edges(keys=True, data=True)))
```

In short, it has every attribute in the summary table, as well as the path coordinates (in pixel space), the nonzero pixel indices, and the image values under the array.

+++

Now, looking at the above skeleton, we notice:
- single branches that haven't been removed (because the original pruning function wasn't iterative/recursive
- small self-loops that aren't removed because they aren't technically endpoints
- dual loops where one of two paths joining points is "dimmer" than the other

+++

Let's look at the signature of `iteratively_prune_paths`:

```{code-cell} ipython3
from skan import iteratively_prune_paths
help(iteratively_prune_paths)
```

So we need to write a function that will identify the edges that we don't want in the graph. To repeat our previous conditions:

- edge branches — one of the endpoints' degrees should be 1.
- self-loops *other* than the final self-loop, which of course we want to keep! 😅
- the "dimmer" edge of a multi-edge pair.

Let's try:

```{code-cell} ipython3
def unwanted(mg, e):
    u, v, k = e
    # first the easy one: the branch is an endpoint
    if mg.degree(u) == 1 or mg.degree(v) == 1:
        return True
    # next, self-loops, other than the final self-loop
    if u == v and len(mg.edges()) > 1:
        return True
    # finally, the dimmer of two of the same edge.
    # We'll use a helper function, 'get_multiedge', that returns
    # a sibling multiedge if it exists and None otherwise
    if (e2 := get_multiedge(mg, e)) is not None:
        # if there is a multiedge, we discard current edge if it's
        # dimmer (lower mean pixel value) than its sibling edge
        return mg.edges[e]['mean_pixel_value'] < mg.edges[e2]['mean_pixel_value']
    return False


def get_multiedge(mg, e):
    u, v, k = e
    edge_keys = set(mg[u][v])  # g[u][v] returns a view of the keys
    if len(edge_keys) > 1:  # multiedge
        other_key = (edge_keys - {k}).pop()
        return (u, v, other_key)
```

```{code-cell} ipython3
skeletons_pruned_iteratively = list(iteratively_prune_paths(nxskel, discard=unwanted))
```

```{code-cell} ipython3
images = np.asarray([sk.skeleton_image for sk in skeletons_pruned_iteratively])
```

```{code-cell} ipython3
def pts_list_to_nd(list_of_coord_arrays):
    arrs = []
    for i, arr in enumerate(list_of_coord_arrays):
        prefix = np.full((arr.shape[0], 1), i)
        arrs.append(np.concatenate((prefix, arr), axis=1))
    return np.concatenate(arrs, axis=0)
```

```{code-cell} ipython3
from skan import summarize

summaries = [summarize(sk, separator='_') for sk in skeletons_pruned_iteratively]
src_pts = [np.asarray(s[['image_coord_src_0', 'image_coord_src_1']]) for s in summaries]
src_pts_nd = pts_list_to_nd(src_pts)
dst_pts = [np.asarray(s[['image_coord_dst_0', 'image_coord_dst_1']]) for s in summaries]
dst_pts_nd = pts_list_to_nd(dst_pts)

all_pts = np.concatenate([src_pts_nd, dst_pts_nd], axis=0)
all_pts_unique = np.unique(all_pts, axis=0)
```

```{code-cell} ipython3
viewer, layer = napari.imshow(images)
```

```{code-cell} ipython3
pts_layer = viewer.add_points(all_pts_unique, size=1, face_color='red')
```

```{code-cell} ipython3
skel0 = skeletons_pruned_iteratively[0]
```

```{code-cell} ipython3
all_paths = [skel0.path_coordinates(i) for i in range(skel0.n_paths)]
paths_table = summaries[0]
paths_table['path_id'] = np.arange(skel0.n_paths)
paths_table['random_path_id'] = np.random.default_rng().permutation(skel0.n_paths)
```

```{code-cell} ipython3
len(np.unique(paths_table['random_path_id']))
```

```{code-cell} ipython3
skel0.n_paths
```

```{code-cell} ipython3
skeleton_layer = viewer.add_shapes(
        all_paths,
        shape_type='path',
        features=paths_table,
        edge_width=0.5,
        edge_color='random_path_id',
        edge_colormap='tab10',
        )
```

```{code-cell} ipython3
skeleton_layer.edge_color_cycle = 'tab10'
```

```{code-cell} ipython3
len(skeleton_pruned_iteratively.nodes())
```

```{code-cell} ipython3
len(skeleton_pruned_iteratively.edges())
```

```{code-cell} ipython3
from skan.csr import nx_to_skeleton

skel_pruned2 = nx_to_skeleton(skeleton_pruned_iteratively)
```

```{code-cell} ipython3
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
internal loops, but we will use the `iteratively_prune_paths()` function to tidy this up

```{code-cell} ipython3
skeleton_smoothed_pruned_iteratively = iteratively_prune_paths(skeleton).skeleton_image
plt.imshow(skeleton_smoothed_pruned_iteratively)
```

We now have a single path for the skeleton which represents the loop of DNA.
