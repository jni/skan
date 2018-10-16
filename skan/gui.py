import os
import json
import asyncio
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
import click


from . import pre, pipe, draw, io, __version__


@asyncio.coroutine
def _async(coroutine, *args):
    loop = asyncio.get_event_loop()
    return (yield from loop.run_in_executor(None, coroutine, *args))


STANDARD_MARGIN = (3, 3, 12, 12)


class Launch(tk.Tk):
    def __init__(self, params_dict=None):
        super().__init__()
        self.title('Skeleton analysis tool')
        self.crop_radius = tk.IntVar(value=0, name='Crop radius')
        self.smooth_method = tk.StringVar(value='Gaussian',
                                          name='Smoothing method')
        self.smooth_method._choices = pre.SMOOTH_METHODS
        self.smooth_radius = tk.DoubleVar(value=0.1, name='Smoothing radius')
        self.threshold_radius = tk.DoubleVar(value=50e-9,
                                             name='Threshold radius')
        self.brightness_offset = tk.DoubleVar(value=0.075,
                                              name='Brightness offset')
        self.image_format = tk.StringVar(value='auto',
                                         name='Image format')
        self.scale_metadata_path = tk.StringVar(value='Scan/PixelHeight',
                                                name='Scale metadata path')
        self.preview_skeleton_plots = tk.BooleanVar(value=True, name='Live '
                                                    'preview skeleton plot?')
        self.save_skeleton_plots = tk.BooleanVar(value=True,
                                                 name='Save skeleton plot?')
        self.skeleton_plot_prefix = tk.StringVar(value='skeleton-plot-',
                                         name='Prefix for skeleton plots')
        self.output_filename = tk.StringVar(value='skeleton.xlsx',
                                            name='Output filename')
        self.parameters = [
            self.crop_radius,
            self.smooth_method,
            self.smooth_radius,
            self.threshold_radius,
            self.brightness_offset,
            self.image_format,
            self.scale_metadata_path,
            self.preview_skeleton_plots,
            self.save_skeleton_plots,
            self.skeleton_plot_prefix,
            self.output_filename,
        ]

        self.input_files = []
        self.output_folder = None

        if params_dict is None:
            params_dict = {}
        self.params_dict = params_dict.copy()
        self.parameter_config(params_dict)

        # allow resizing
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.create_main_frame()

    def parameter_config(self, params_dict):
        """Set parameter values from a config dictionary."""
        if isinstance(params_dict, str):
            if params_dict.startswith('{'):  # JSON string
                params_dict = json.loads(params_dict)
            else:  # config file
                with open(params_dict) as params_fin:
                    params_dict = json.load(params_fin)
            self.params_dict.update(params_dict)
        name2param = {p._name.lower(): p for p in self.parameters}
        for param, value in self.params_dict.items():
            if param.lower() in name2param:
                name2param[param].set(value)
                params_dict.pop(param)
        for param, value in params_dict.copy().items():
            if param.lower() == 'input files':
                self.input_files = value
                params_dict.pop(param)
            elif param.lower() == 'output folder':
                self.output_folder = Path(os.path.expanduser(value))
                params_dict.pop(param)
            elif param.lower() == 'version':
                print(f'Parameter file version: {params_dict.pop(param)}')
        for param in params_dict:
            print(f'Parameter not recognised: {param}')

    def save_parameters(self, filename=None):
        out = {p._name.lower(): p.get() for p in self.parameters}
        out['input files'] = self.input_files
        out['output folder'] = str(self.output_folder)
        out['version'] = __version__
        if filename is None:
            return json.dumps(out)
        attempt = 0
        base, ext = os.path.splitext(filename)
        while os.path.exists(filename):
            filename = f'{base} ({attempt}){ext}'
            attempt += 1
        with open(filename, mode='wt') as fout:
            json.dump(out, fout, indent=2)

    def create_main_frame(self):
        main = ttk.Frame(master=self, padding=STANDARD_MARGIN)
        main.grid(row=0, column=0, sticky='nsew')
        self.create_parameters_frame(main)
        self.create_buttons_frame(main)
        main.pack()

    def create_parameters_frame(self, parent):
        parameters = ttk.Frame(master=parent, padding=STANDARD_MARGIN)
        parameters.grid(sticky='nsew')

        heading = ttk.Label(parameters, text='Analysis parameters')
        heading.grid(column=0, row=0, sticky='n')

        for i, param in enumerate(self.parameters, start=1):
            param_label = ttk.Label(parameters, text=param._name)
            param_label.grid(row=i, column=0, sticky='nsew')
            if type(param) == tk.BooleanVar:
                param_entry = ttk.Checkbutton(parameters, variable=param)
            elif hasattr(param, '_choices'):
                param_entry = ttk.OptionMenu(parameters, param, param.get(),
                                             *param._choices.keys())
            else:
                param_entry = ttk.Entry(parameters, textvariable=param)
            param_entry.grid(row=i, column=1, sticky='nsew')

    def create_buttons_frame(self, parent):
        buttons = ttk.Frame(master=parent, padding=STANDARD_MARGIN)
        buttons.grid(sticky='nsew')
        actions = [
            ('Choose config', self.choose_config_file),
            ('Choose files', self.choose_input_files),
            ('Choose output folder', self.choose_output_folder),
            ('Run', lambda: asyncio.ensure_future(self.run()))
        ]
        for col, (action_name, action) in enumerate(actions):
            button = ttk.Button(buttons, text=action_name,
                                command=action)
            button.grid(row=0, column=col)

    def choose_config_file(self):
        config_file = tk.filedialog.askopenfilename()
        self.parameter_config(config_file)

    def choose_input_files(self):
        self.input_files = tk.filedialog.askopenfilenames()
        if len(self.input_files) > 0 and self.output_folder is None:
            self.output_folder = Path(os.path.dirname(self.input_files[0]))

    def choose_output_folder(self):
        self.output_folder = Path(
                tk.filedialog.askdirectory(initialdir=self.output_folder))

    def make_figure_window(self):
        self.figure_window = tk.Toplevel(self)
        self.figure_window.wm_title('Preview')
        screen_dpi = self.figure_window.winfo_fpixels('1i')
        screen_width = self.figure_window.winfo_screenwidth()  # in pixels
        figure_width = screen_width / 2 / screen_dpi
        figure_height = 0.75 * figure_width
        self.figure = Figure(figsize=(figure_width, figure_height),
                             dpi=screen_dpi)
        ax0 = self.figure.add_subplot(221)
        axes = [self.figure.add_subplot(220 + i, sharex=ax0, sharey=ax0)
                for i in range(2, 5)]
        self.axes = np.array([ax0] + axes)
        canvas = FigureCanvasTkAgg(self.figure, master=self.figure_window)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, self.figure_window)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    async def run(self):
        print('Input files:')
        for file in self.input_files:
            print('  ', file)
        print('Parameters:')
        for param in self.parameters:
            p = param.get()
            print('  ', param, type(p), p)
        print('Output:', self.output_folder)
        save_skeleton = ('' if not self.save_skeleton_plots.get() else
                         self.skeleton_plot_prefix.get())
        images_iterator = pipe.process_images(
                self.input_files, self.image_format.get(),
                self.threshold_radius.get(),
                self.smooth_radius.get(),
                self.brightness_offset.get(),
                self.scale_metadata_path.get(),
                crop_radius=self.crop_radius.get(),
                smooth_method=self.smooth_method.get())
        if self.preview_skeleton_plots.get():
            self.make_figure_window()
        elif self.save_skeleton_plots.get():
            self.figure = plt.figure()
            ax0 = self.figure.add_subplot(221)
            axes = [self.figure.add_subplot(220 + i, sharex=ax0, sharey=ax0)
                    for i in range(2, 5)]
            self.axes = np.array([ax0] + axes)
        self.save_parameters(self.output_folder / 'skan-config.json')
        for i, result in enumerate(images_iterator):
            if i < len(self.input_files):
                filename, image, thresholded, skeleton, framedata = result
                if save_skeleton:
                    for ax in self.axes:
                        ax.clear()
                    w, h = draw.pixel_perfect_figsize(image)
                    self.figure.set_size_inches(4*w, 4*h)
                    draw.pipeline_plot(image, thresholded, skeleton, framedata,
                                       figure=self.figure, axes=self.axes)
                    output_basename = (save_skeleton +
                                       os.path.basename(
                                           os.path.splitext(filename)[0]) +
                                       '.png')
                    output_filename = str(self.output_folder / output_basename)
                    self.figure.savefig(output_filename)
                if self.preview_skeleton_plots.get():
                    self.figure.canvas.draw_idle()
            else:
                result_full, result_image = result
                result_filtered = result_full[(result_full['mean shape index']>0.125) &
                                              (result_full['mean shape index']<0.625) &
                                              (result_full['branch-type'] == 2) &
                                              (result_full['euclidean-distance']>0)]
                ridgeydata = result_filtered.groupby('filename')[['filename','branch-distance','scale','euclidean-distance','squiggle','mean shape index']].mean()
                io.write_excel(str(self.output_folder /
                                   self.output_filename.get()),
                               branches=result_full,
                               images=result_image,
                               filtered=ridgeydata,
                               parameters=json.loads(self.save_parameters()))


def tk_update(loop, app):
    try:
        app.update()
    except tkinter.TclError:
        loop.stop()
        return
    loop.call_later(.01, tk_update, loop, app)


@click.command()
@click.option('-c', '--config', default='',
              help='JSON configuration file.')
def launch(config):
    params = json.load(open(config)) if config else None
    app = Launch(params)
    loop = asyncio.get_event_loop()
    tk_update(loop, app)
    loop.run_forever()
