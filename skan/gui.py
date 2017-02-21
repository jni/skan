import os
import json
import asyncio
import concurrent.futures
import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
import click


from . import pre, pipe, io, __version__
from . import pipe, io, __version__


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
        self.save_skeleton_plots = tk.BooleanVar(value=True,
                                                 name='Save skeleton plot?')
        self.skeleton_plot_prefix = tk.StringVar(value='skeleton-plot-',
                                         name='Prefix for skeleton plots')
        self.output_filename = tk.StringVar(value='skeleton.xslx',
                                            name='Output filename')
        self.parameters = [
            self.crop_radius,
            self.smooth_method,
            self.smooth_radius,
            self.threshold_radius,
            self.brightness_offset,
            self.image_format,
            self.scale_metadata_path,
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
                self.output_folder = os.path.expanduser(value)
                params_dict.pop(param)
            elif param.lower() == 'version':
                print(f'Parameter file version: {params_dict.pop(param)}')
        for param in params_dict:
            print(f'Parameter not recognised: {param}')

    def save_parameters(self, filename):
        out = {p._name.lower(): p.get() for p in self.parameters}
        out['input files'] = self.input_files
        out['output folder'] = self.output_folder
        out['version'] = __version__
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
            self.output_folder = os.path.dirname(self.input_files[0])

    def choose_output_folder(self):
        self.output_folder = \
                tk.filedialog.askdirectory(initialdir=self.output_folder)

    @asyncio.coroutine
    def run(self):
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
        result_full, result_image = yield from _async(pipe.process_images,
                self.input_files, self.image_format.get(),
                self.threshold_radius.get(),
                self.smooth_radius.get(),
                self.brightness_offset.get(),
                self.scale_metadata_path.get(),
                save_skeleton,
                self.output_folder,
                crop_radius=self.crop_radius.get(),
                smooth_method=self.smooth_method.get())
        io.write_excel(self.output_filename.get(),
                       branches=result_full,
                       images=result_image,
                       parameters=json.loads(self.save_parameters()))
        self.save_parameters(os.path.join(self.output_folder,
                                          'skan-config.json'))


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
