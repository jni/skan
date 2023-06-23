# skan v0.11.1

This is a bugfix release. It adds napari.yaml to the manifest so that it
is correctly shipped with skan, and it fixes compatibility issues with
more recent versions of openpyxl. With thanks to
[James Ryan](https://github.com/jamesyan-git)!

## Bug fixes

- [#202](https://github.com/jni/skan/pull/202): 🐞 Bug Fix: Include napari.yaml in build
- [#203](https://github.com/jni/skan/pull/203): Remove deprecated code from `io.py`

# skan v0.11.0

This release of skan incorporates several bug fixes, new API features, and
documentation improvements. It also finalizes an API change started in 0.10.0:
now junction points are always resolved using a minimum spanning tree, and the
`uniquify_junctions` and `junction_mode` arguments to `Skeleton` are
deprecated (see our [FAQ](../faq)). Finally, this is the first release
containing a [napari plugin](https://napari.org/stable/plugins/index.html)!
Currently all it does is skeletonize a Labels layer, but this is just the
beginning for GUI-based skeleton analysis.

Thanks to everyone who has helped make this release possible, including
[Kushaan Gupta](https://github.com/kushaangupta), [Lucy
Liu](https://github.com/lucyleeow), [Ryan Ly](https://github.com/rly),
[James Ryan](https://github.com/jamesyan-git), and
[Simon Savary](https://github.com/ssavary)! Not to speak of all the
contributors who make our upstream libraries possible! 🙏

## API changes

- [#143](https://github.com/jni/skan/pull/143): the `unique_junctions` and
  `junction_mode` keyword arguments are removed. Junctions are always resolved
  by finding the minimum spanning tree of the junction pixels. This PR also
  speeds up building of the pixel graph.

## New features

- [#150](https://github.com/jni/skan/pull/150),
  [#164](https://github.com/jni/skan/pull/164): add Sholl analysis. (Thanks to
  [Kushaan Gupta](https://github.com/kushaangupta) for the collaboration that
  led to this feature!)
- [#184](https://github.com/jni/skan/pull/184): add napari plugin.

## Bug fixes

- [#152](https://github.com/jni/skan/pull/152): Some pixel graphs had missing
  paths in their skeletons because of a mistake in how the graphs were
  traversed. Thanks [Simon Savary](https://github.com/ssavary) for the detailed
  report that led to the fix! ([#147](https://github.com/jni/skan/issues/147))
- [#193](https://github.com/jni/skan/pull/193),
  [#183](https://github.com/jni/skan/pull/183): fix the calculation of the
  buffer size needed for the pixel path graph in the presence of 0-degree
  nodes (isolated pixels).
- [#135](https://github.com/jni/skan/pull/135): the `unique_junctions` keyword
  argument to the Skeleton class is deprecated. Use instead `junction_mode`.
  Note however that this option will be removed in 0.11, so you should pin your
  skan dependency if you need this behavior.
- [#139](https://github.com/jni/skan/pull/139): the skan GUI and corresponding
  skan.gui module and skan command have all been removed. A new, much more
  sophisticated napari plugin is in development at
  https://github.com/kevinyamauchi/napari-skeleton-curator and will be folded
  into a future version of skan (probably v0.11).

## Documentation

- [#155](https://github.com/jni/skan/pull/155),
  [#156](https://github.com/jni/skan/pull/156),
  [#159](https://github.com/jni/skan/pull/159): Add documentation on 3D display
  of skeletons in napari.
- [#173](https://github.com/jni/skan/pull/173),
  [#175](https://github.com/jni/skan/pull/),
  [#177](https://github.com/jni/skan/pull/177): support multiple versions of
  documentation. (!) (This series of PRs in particular is close to my heart
  because deprecations and API changes like those listed above are much more
  painful if the old versions are just *erased*! Thanks to [Lucy
  Liu](https://github.com/lucyleeow) for her efforts and expertise here!)
- [#194](https://github.com/jni/skan/pull/194),
  [#195](https://github.com/jni/skan/pull/195): overhaul of documentation and
  build infrastructure.

## Misc

- [#167](https://github.com/jni/skan/pull/167): drop Python 3.7 support.
- [#188](https://github.com/jni/skan/pull/188),
  [#189](https://github.com/jni/skan/pull/189),
  [#190](https://github.com/jni/skan/pull/190): update requirements.

