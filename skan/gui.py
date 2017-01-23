import os
import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk


from . import pipe


STANDARD_MARGIN = (3, 3, 12, 12)


class Launch(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Skeleton analysis tool')
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
        self.parameters = [
            self.smooth_radius,
            self.threshold_radius,
            self.brightness_offset,
            self.image_format,
            self.scale_metadata_path,
            self.save_skeleton_plots
        ]

        self.input_files = []
        self.output_folder = os.path.expanduser('~/Desktop')

        # allow resizing
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.create_main_frame()

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
            else:
                param_entry = ttk.Entry(parameters, textvariable=param)
            param_entry.grid(row=i, column=1, sticky='nsew')

    def create_buttons_frame(self, parent):
        buttons = ttk.Frame(master=parent, padding=STANDARD_MARGIN)
        buttons.grid(sticky='nsew')
        actions = [
            ('Choose files', self.choose_input_files),
            ('Choose output folder', self.choose_output_folder),
            ('Run', self.run)
        ]
        for col, (action_name, action) in enumerate(actions):
            button = ttk.Button(buttons, text=action_name,
                                command=action)
            button.grid(row=0, column=col)

    def choose_input_files(self):
        self.input_files = tk.filedialog.askopenfilenames()

    def choose_output_folder(self):
        self.output_folder = \
                tk.filedialog.askdirectory(initialdir=self.output_folder)

    def run(self):
        print('Input files:')
        for file in self.input_files:
            print('  ', file)
        print('Parameters:')
        for param in self.parameters:
            p = param.get()
            print('  ', param, type(p), p)
        print('Output:', self.output_folder)
        result = pipe.process_images(self.input_files, self.image_format.get(),
                                     self.threshold_radius.get(),
                                     self.smooth_radius.get(),
                                     self.brightness_offset.get(),
                                     self.scale_metadata_path.get(),
                                     self.save_skeleton_plots.get(),
                                     self.output_folder)
        result.to_csv(os.path.join(self.output_folder, 'skeletons.csv'))


def launch():
    app = Launch()
    app.mainloop()
