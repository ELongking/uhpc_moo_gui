import tkinter as tk
from tkinter import messagebox as mb
from tkinter import ttk
import pandas as pd
from tkinter import filedialog
from filter import DATA
from multi_optimization import main


def predict_template(frame):
    result_dict = {'data': '', 'export_path': '', 'custom_bound': ''}
    import_path, export_path, bound_path = tk.StringVar(), tk.StringVar(), tk.StringVar()

    def read_data():
        path = filedialog.askopenfilename(title=r'Choose data file')
        path = path.replace("/", "\\\\")
        import_path.set(path)

        if path[-3:] == 'csv':
            data = pd.read_csv(path)
            result_dict['data'] = data
        elif path[-4:] == 'xlsx':
            data = pd.read_excel(path)
            result_dict['data'] = data
        else:
            mb.showwarning('warning', 'you should import .csv or .xlsx file')

    def save_data():
        path = filedialog.askdirectory(title=r'choose where you save the log and model file')
        path = path.replace("/", "\\\\")
        result_dict['export_path'] = path
        export_path.set(path)

    def custom_bound():
        path = filedialog.askopenfilename(
            title=r'if the distribution of several tabular is different, plz firstly import your bound file here')
        if path[-4:] != 'xlsx':
            mb.showwarning('warning', 'you should import .xlsx file')
        path = path.replace("/", "\\\\")
        result_dict['custom_bound'] = path
        bound_path.set(path)

    def refresh():
        if import_path.get() == '' or export_path.get() == '':
            generate_button['state'] = tk.DISABLED
        else:
            generate_button['state'] = tk.NORMAL
        return frame.after(1000, refresh)

    data_button = tk.Button(frame, text='import file (.csv or .xlsx)', command=read_data)
    data_button.grid(row=0, column=0, padx=20, pady=10)
    data_entry = tk.Entry(frame, width=40, textvariable=import_path)
    data_entry.grid(row=0, column=1, padx=20, pady=10)

    export_button = tk.Button(frame, text='export directory', command=save_data)
    export_button.grid(row=1, column=0, padx=20, pady=10)
    export_entry = tk.Entry(frame, width=40, textvariable=export_path)
    export_entry.grid(row=1, column=1, padx=20, pady=10)

    addition_button = tk.Button(frame, text='custom bound (.xlsx)', command=custom_bound)
    addition_button.grid(row=2, column=0, padx=20, pady=10)
    addition_entry = tk.Entry(frame, width=40, textvariable=bound_path)
    addition_entry.grid(row=2, column=1, padx=20, pady=10)

    generate_button = tk.Button(frame, text='start',
                                command=lambda:
                                DATA(result_dict['data'],
                                     result_dict['export_path'],
                                     bound_path=result_dict['custom_bound']).main())
    generate_button.grid(row=3, column=0, padx=10, pady=10)
    refresh()


def optimized_template(frame):
    model_file, constraint_file, export_file = tk.StringVar(), tk.StringVar(), tk.StringVar()
    result_dict = {'model_file': [], 'constraint_file': '', 'export_file': ''}

    def refresh():
        if model_file.get() == [] or constraint_file.get() == '' or export_file.get() == '':
            start_button['state'] = tk.DISABLED
        else:
            start_button['state'] = tk.NORMAL
        return frame.after(1000, refresh)

    def read_model():
        path = filedialog.askopenfilename(title=r'Choose data file')
        path = path.replace("/", "\\\\")
        if path[-3:] != 'pkl':
            mb.showwarning('warning', 'you should import .pkl file')
        model_listbox.insert('end', path)
        result_dict['model_file'].append(path)

    def read_constraint():
        path = filedialog.askopenfilename(title=r'Choose constraint file')
        path = path.replace("/", "\\\\")
        if path[-3:] != 'txt':
            mb.showwarning('warning', 'you should import .txt file')
        result_dict['constraint_file'] = path
        constraint_file.set(path)

    def save_data():
        path = filedialog.askdirectory(title=r'choose where you save the log and model file')
        path = path.replace("/", "\\\\")
        result_dict['export_file'] = path
        export_file.set(path)

    def remove_model():
        index = model_listbox.curselection()
        model_listbox.delete(index)
        result_dict['model_file'].pop(index)

    model_button = tk.Button(frame, text='model file (.pkl)', command=read_model)
    model_button.grid(row=0, column=0, padx=20, pady=5)
    model_listbox = tk.Listbox(frame, height=5, width=40, listvariable=model_file)
    model_listbox.grid(row=0, column=1, padx=20, pady=5)

    model_delete_button = tk.Button(frame, text='remove', command=remove_model)
    model_delete_button.grid(row=0, column=2, padx=10, pady=5)

    limit_button = tk.Button(frame, text='constraint file (.txt)', command=read_constraint)
    limit_button.grid(row=1, column=0, padx=20, pady=5)
    limit_entry = tk.Entry(frame, width=40, textvariable=constraint_file)
    limit_entry.grid(row=1, column=1, padx=20, pady=5)

    generate_button = tk.Button(frame, text='result directory (.xlsx)', command=save_data)
    generate_button.grid(row=2, column=0, padx=20, pady=5)
    generate_entry = tk.Entry(frame, width=40, textvariable=export_file)
    generate_entry.grid(row=2, column=1, padx=20, pady=5)

    start_button = tk.Button(frame, text='start',
                             command=lambda: main(result_dict['model_file'], result_dict['constraint_file'],
                                                  result_dict['export_file']))
    start_button.grid(row=3, column=0, padx=20, pady=10)
    refresh()


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('UHPC mixture ratio design')
        self.root.geometry('800x600')
        self._initialize()

    def _initialize(self):
        self.notebook = ttk.Notebook(self.root)
        self.predict = tk.Frame(self.notebook)
        self.optimize = tk.Frame(self.notebook)
        self.notebook.add(self.predict, text='Predict model (Step 1)', padding=20)
        self.notebook.add(self.optimize, text='MOO (Step 2)', padding=20)
        self.notebook.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        predict_template(self.predict)
        optimized_template(self.optimize)

    def open(self):
        self.root.mainloop()


ROOT = GUI()
ROOT.open()
