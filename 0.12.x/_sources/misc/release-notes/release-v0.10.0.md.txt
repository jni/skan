# skan v0.10.0

This is a major release of skan that changes, removes, or deprecates much
functionality. As skan has grown in popularity, we've been working hard to
clean out the warts and kinks in the API, improve compatibility with libraries
such as dask, and fix several bugs reported by users. This has brought one
major change in how skan computes branch lengths
([#135](https://github.com/jni/skan/pull/135)): junctions are now cleaned up by
computing their minimum spanning tree rather than by computing their centroid
(see the [FAQ](https://jni.github.io/skan/faq.html)). This change can be
reverted with a keyword argument in this version (`junction_mode='centroid'`),
but **will be mandatory in upcoming versions**. If you need to preserve the old
results, pin skan to <v0.11.

Thanks to [Genevieve Buckley](https://github.com/GenevieveBuckley), [Marianne
Corvellec](https://github.com/mkcor), [Zoltan
Csati](https://github.com/CsatiZoltan), [Marlene da Vitoria
Lobo](https://github.com/marlenedavitoria), and [Kevin
Yamauchi](https://github.com/kevinyamauchi) for their contributions!

## API changes

- [#135](https://github.com/jni/skan/pull/135): the `unique_junctions` keyword
  argument to the Skeleton class is deprecated. Use instead `junction_mode`.
  Note however that this option will be removed in 0.11, so you should pin your
  skan dependency if you need this behavior.
- [#139](https://github.com/jni/skan/pull/139): the skan GUI and corresponding
  skan.gui module and skan command have all been removed. A new, much more
  sophisticated napari plugin is in development at
  https://github.com/kevinyamauchi/napari-skeleton-curator and will be folded
  into a future version of skan (probably v0.11).

## Improvements

- skan tests now pass on GitHub Actions on all platforms
  ([#139](https://github.com/jni/skan/pull/139)).
- skan documentation is now built and deployed on GitHub Actions
  ([#140](https://github.com/jni/skan/pull/140)).
- skan releases are created using GitHub Actions
  ([#141](https://github.com/jni/skan/pull/141)).
- the skan code base is now formatted by yapf
  ([#136](https://github.com/jni/skan/pull/136)).
- skan is now easier to adapt for dask arrays (though there is still much work
  to be done here) (([#107](https://github.com/jni/skan/pull/107),
  [#112](https://github.com/jni/skan/pull/112) and
  [#123](https://github.com/jni/skan/pull/123)).
