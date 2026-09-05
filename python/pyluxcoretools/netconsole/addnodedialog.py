import tkinter as tk
from tkinter import ttk


class AddNodeDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Add a new rendering node")
        self.geometry("400x150")
        self.transient(parent)
        self.grab_set()

        # Main layout
        self.grid_layout = ttk.Frame(self, padding="10")
        self.grid_layout.pack(fill=tk.BOTH, expand=True)

        # Title label
        self.label_title = ttk.Label(
            self.grid_layout,
            text="<B>Rendering Node Configuration<B>",
            justify=tk.CENTER,
        )
        self.label_title.grid(row=0, column=0, columnspan=2, pady=5)

        # Form layout
        self.form_frame = ttk.Frame(self.grid_layout)
        self.form_frame.grid(row=1, column=0, columnspan=2, pady=5)

        # IP Address
        self.label_ip = ttk.Label(
            self.form_frame, text="Hostname or IP address:"
        )
        self.label_ip.grid(row=0, column=0, sticky=tk.E, padx=5, pady=2)

        self.entry_ip = ttk.Entry(self.form_frame, width=20)
        self.entry_ip.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Port
        self.label_port = ttk.Label(self.form_frame, text="Port:")
        self.label_port.grid(row=1, column=0, sticky=tk.E, padx=5, pady=2)

        self.entry_port = ttk.Entry(self.form_frame, width=10)
        self.entry_port.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Spacer
        self.spacer = ttk.Frame(self.grid_layout, height=20)
        self.spacer.grid(row=2, column=0)

        # Button box
        self.button_frame = ttk.Frame(self.grid_layout)
        self.button_frame.grid(row=2, column=1, pady=5)

        self.button_ok = ttk.Button(
            self.button_frame, text="OK", command=self.on_ok
        )
        self.button_ok.pack(side=tk.LEFT, padx=5)

        self.button_cancel = ttk.Button(
            self.button_frame, text="Cancel", command=self.on_cancel
        )
        self.button_cancel.pack(side=tk.LEFT, padx=5)

        # Dialog result
        self.result = None

    def on_ok(self):
        self.result = {
            "ip": self.entry_ip.get(),
            "port": self.entry_port.get(),
        }
        self.destroy()

    def on_cancel(self):
        self.destroy()

    def show(self):
        self.wait_window(self)
        return self.result
