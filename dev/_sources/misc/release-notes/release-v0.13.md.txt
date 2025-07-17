# skan v0.13.0

This is a minor step forward from v0.12.x which adds a new API:
`skan.csr.PathGraph` is a more abstract version of a Skeleton, which only
needs a pixel adjacency matrix to work, rather than a full skeleton image.
The motivations for PathGraph are numerous:

- a simpler dataclass object that doesn't require complex instantiation
  logic. See e.g. Glyph's [Stop Writing `__init__`
  Methods](https://blog.glyph.im/2025/04/stop-writing-init-methods.html)
- making it easier to compute the pixel adjacency matrix separately, for
  example [using dask when the images don't fit in
  memory](https://blog.dask.org/2021/05/07/skeleton-analysis), and having
  an API for which you can provide this matrix (rather than having to
  modify a Skeleton instance in-place).
- allowing more flexible use cases, for example to use skan to measure
  tracks, as in
  [live-image-tracking-tools/traccuracy#251](https://github.com/live-image-tracking-tools/traccuracy/pull/251).

Due to some urgent need to use this code in the wild, this release doesn't
provide any documentation examples. Indeed, the new class may see some changes
in upcoming releases based on user feedback. See the discussion in
[jni/skan#246](https://github.com/jni/skan/pull/246) for details. Look for
further refinement of this idea in the 0.13.x releases!

## New features

- [#246](https://github.com/jni/skan/pull/246): New API: PathGraph dataclass

