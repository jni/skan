Installing Skan
===============

Skan is a pure Python package with several non-Python dependencies, notably
NumPy, SciPy, and Numba. It can be installed from several Python package
sources: PyPI, conda-forge, and GitHub.

Getting the Python package and dependencies
-------------------------------------------

The easiest way to install Skan is to get
`conda <https://conda.io/miniconda.html>`_ and then type the
following into your terminal::

    conda install -c conda-forge skan

It is also available on PyPI::

    pip install skan

Though getting Numba from PyPI might be a bit more challenging. Your mileage
may vary.

To get the bleeding edge version of Skan directly from the GitHub source code,
do::

    pip install git+https://github.com/jni/skan

Finally, to contribute to Skan's development, or to just make modifications for
your own personal use, optionally [fork]() the GitHub repository to your own
GitHub account, clone it, then::

    pip install -e path/to/cloned/skan

This will install your local Skan copy, and any modifications you make to the
code will be reflected the behaviour of the software.
