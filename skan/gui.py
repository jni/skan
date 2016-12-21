import tkinter as tk
from tkinter import ttk


STANDARD_MARGIN = (3, 3, 12, 12)


class Launch(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Skeleton analysis tool')
        self.smooth_radius = tk.DoubleVar(value=0.1, name='Smoothing radius')
        self.threshold_radius = tk.DoubleVar(value=50e-9,
                                             name='Threshold radius')
        self.brightness_offset = tk.DoubleVar(value=-10,
                                              name='Brightness offset')
        self.scale_metadata_path = tk.StringVar(value='Scan,PixelHeight',
                                                name='Scale metadata path')
        self.parameters = [
            self.smooth_radius,
            self.threshold_radius,
            self.brightness_offset,
            self.scale_metadata_path
        ]

        # allow resizing
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.create_main_frame()

    def create_main_frame(self):
        main = ttk.Frame(master=self, padding=STANDARD_MARGIN)
        main.grid(row=0, column=0, sticky='nsew')
        parameters = ttk.Frame(master=main, padding=STANDARD_MARGIN)
        parameters.grid(sticky='nsew')

        cur_row = 0

        heading = ttk.Label(parameters, text='Analysis parameters')
        heading.grid(column=0, row=cur_row, sticky='n')
        cur_row += 1

        for i, param in enumerate(self.parameters):
            param_label = ttk.Label(parameters, text=param._name)
            param_label.grid(row=cur_row + i, column=0, sticky='nsew')
            param_entry = ttk.Entry(parameters, textvariable=param)
            param_entry.grid(row=cur_row + i, column=1, sticky='nsew')
        cur_row += i

        main.pack()


if __name__ == '__main__':
    app = Launch()
    app.mainloop()
