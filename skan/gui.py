import tkinter as tk
from tkinter import ttk


STANDARD_MARGIN = (3, 3, 12, 12)


class Launch(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Skeleton analysis tool')
        self.smooth_radius = tk.DoubleVar(value=0.1, name='foo')
        self.threshold_radius = tk.DoubleVar(value=50e-9, name='bar')
        self.brightness_offset = tk.DoubleVar(value=-10, name='baz')
        self.scale_metadata_path = tk.StringVar(value='Scan,PixelHeight',
                                                name='bork')

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

        smooth_label = ttk.Label(parameters, text='Smoothing radius')
        smooth_label.grid(column=0, row=cur_row, sticky='nsew')
        smooth = ttk.Entry(parameters, textvariable=self.smooth_radius)
        smooth.grid(column=1, row=cur_row, sticky='nsew')
        cur_row += 1

        threshold_label = ttk.Label(parameters, text='Threshold radius')
        threshold_label.grid(row=cur_row, column=0, sticky='nsew')
        threshold = ttk.Entry(parameters, textvariable=self.threshold_radius)
        threshold.grid(row=cur_row, column=1, sticky='nsew')
        cur_row += 1

        main.pack()


if __name__ == '__main__':
    app = Launch()
    app.mainloop()
