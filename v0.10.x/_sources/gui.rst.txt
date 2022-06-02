Using Skan's GUI
================

Launching the GUI
-----------------

To launch Skan's graphical user interface after installation, run the command::

    skan-gui

in your terminal. This will launch the configuration interface, which looks
like this:

.. image:: _static/gui-screenshot.png
   :width: 640px
   :alt: Skan pipeline configuration GUI
   :align: center

Interpreting the GUI
--------------------

The GUI is a simple launcher for the :any:`skan.pipe.process_images` function.
This function runs the following analysis for each image:

- Crop the image
- Smooth/denoise it
- Threshold it
- Skeletonise it
- Compute statistics for the resulting skeleton
- Compile the statistics for all the images into a single table
- Save the results to an Excel file

The parameters relate to this pipeline.

- *Crop radius:* the number of pixels to remove on each side of the input image
  before analysis.
- *Smoothing method:* the denoising method to apply to the image before
  thresholding. Defaults to Gaussian smoothing, but total variation and
  non-local means denoising are also available.
- *Smoothing radius:* the radius for smoothing *as a proportion of the
  threshold radius (below)*.
- *Threshold radius:* the radius for thresholding the smoothed image, in real
  units. (i.e. in the shown, this is 50nm, a reasonable radius to encompass the
  width of a spectrin molecule). This should be chosen as the radius of the
  smallest area containing both foreground and background pixels. If this is
  set to 0, a global Otsu thresholding method is used. For values greater than
  0, Sauvola thresholding is used.
- *Brightness offset:* This corresponds to the *k* parameter in the Sauvola
  equation (see [here]()), or to a value subtracted from the Otsu-computed
  threshold before thresholding.
- *Image format:* The file format of the input images. This corresponds to the
  ``format`` keyword argument in the ``imageio.imread`` function.
- *Scale metadata path:* The location in the metadata hierarchy of the pixel
  scale. The default is correct for FEI electron microscopes
  (``format='fei'``).
- *Live preview skeleton plot?:* whether to show the skeletonisation process for
  each image as it is processed. This incurs a performance penalty of about 2x
  but can save time in cases where the parameter choice is incorrect and many
  images are being processed.
- *Save skeleton plot?:* Whether to save skeletonisation plots next to the
  output results. There are useful for verifation after a run.
- *Prefix for skeleton plots:* Skeleton plots will be saved as
  ``<prefix><input-filename-without-extension>.png``.
- *Output filename:* self-explanatory.
- *Choose config:* Load parameters from a configuration file. These files are
  saved by Skan with every GUI run. The loaded parameters can be modified
  before starting a new run.
- *Choose files:* Choose the input files.
- *Choose output folder:* Select where the output files will be saved.
- *Run:* Start the pipeline run.

Configuration files
-------------------

If you want to repeat a previous analysis, you can point it to an old
configuration file (which Skan writes automatically after each GUI run)::

    skan-gui -c path/to/skan-config.json

This will automatically set all the previous parameters of the GUI, including
the input file selection.

You can also choose a configuration file from the GUI itself.

Configuration files are useful for repeating largely identical analyses with
one or two parameter changes, or for repeating an analysis with fixed
parameters but different input files.
