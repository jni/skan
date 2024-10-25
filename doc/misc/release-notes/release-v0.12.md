# skan v0.12.0

This release adds NumPy 2.0 compatibility (while remaining compatible with 1.x)
([#229](https://github.com/jni/skan/pull/229)). It also lays the groundwork for
new skeleton editing features with bidirectional Skeleton to NetworkX
conversion functions ([#224](https://github.com/jni/skan/pull/224).

We also have a minor deprecation that should improve quality of life in the
future: column names in the summary dataframe can now use `_` as the separator
(instead of `-`), which allows one to use the pandas attribute access for
columns (for example, `summary.branch_distance` instead of
`summary['branch-distance']`. Use the `separator='_'` keyword argument to
`summarize` to take advantage of this feature (which will become the default in
a future version), or `separator='-'` to maintain the current behavior even
when new versions come out ([#215](https://github.com/jni/skan/pull/215)).

The napari plugin now lets you make a Shapes layer fully backed by a Skeleton
dataset, including coloring the edges by features in the summary table
([#201](https://github.com/jni/skan/pull/201)).

Thanks to [Neil Shephard](https://github.com/ns-rse),
[James Ryan](https://github.com/jamesyan-git),
[Jarod Hanko](https://github.com/jarodhanko-crafco), and
[Tim Monko](https://github.com/TimMonko) for their contributions to this
release! 🙏 You can find the full list of changes below:

## API changes

- [#215](https://github.com/jni/skan/pull/215): The separators used for column
  names are configurable, and will transition to `_` in the future. This is to
  make it easier to use the dataframe attribute interface, e.g.
  `summary.branch_distance`

## New features

- [#229](https://github.com/jni/skan/pull/229): NumPy 2 compatibility
- [#224](https://github.com/jni/skan/pull/224): Create a networkx summary graph
  from a Skeleton
- [#201](https://github.com/jni/skan/pull/201): Add napari widget to generate
  shapes layer from a skeletonized label layer

## Improvements

- [#220](https://github.com/jni/skan/pull/220): Allow mean pixel value
  calculation from integer values, not just floats
- [#212](https://github.com/jni/skan/pull/212): Improved error reporting and
  tests for prune_paths methods

## Bug fixes

- [#221](https://github.com/jni/skan/pull/221): Fix documentation builds
- [#210](https://github.com/jni/skan/pull/210): Cache skeleton_image shape for
  use by the path_label_image method

## Documentation

- [#231](https://github.com/jni/skan/pull/231): Add 0.12 release notes

## Misc

- [#232](https://github.com/jni/skan/pull/232): Use python -m build for wheel
  and sdist
- [#218](https://github.com/jni/skan/pull/218): Fix pyproject.toml metadata
  formatting
- [#217](https://github.com/jni/skan/pull/217): Migrate from setup.cfg to
  pyproject.toml

