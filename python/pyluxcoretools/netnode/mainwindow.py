###############################################################################
# Copyright 1998-2025 by authors (see AUTHORS.txt)
#
#   This file is part of LuxCoreRender.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import tkinter as tk
from tkinter import ttk


class MainWindow(tk.Tk):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.root.title("PyLuxCore Tool NetNode")
        self.root.geometry("600x600")

        # Central widget
        self.central_frame = ttk.Frame(self.root, padding="10")
        self.central_frame.pack(fill=tk.BOTH, expand=True)

        # Node config frame
        self.frame_node_config = ttk.LabelFrame(
            self.central_frame,
            text="Rendering Node Configuration",
            padding="10",
        )
        self.frame_node_config.pack(fill=tk.X, pady=5)

        # Form layout
        self.form_frame = ttk.Frame(self.frame_node_config)
        self.form_frame.pack(fill=tk.X)

        # Labels and entries
        ttk.Label(self.form_frame, text="Host name or IP address:").grid(
            row=0, column=0, sticky=tk.E, pady=2
        )
        self.entry_ip = ttk.Entry(self.form_frame, width=30)
        self.entry_ip.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(self.form_frame, text="Port:").grid(
            row=1, column=0, sticky=tk.E, pady=2
        )
        self.entry_port = ttk.Entry(self.form_frame, width=15)
        self.entry_port.grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(self.form_frame, text="Broadcast address:").grid(
            row=2, column=0, sticky=tk.E, pady=2
        )
        self.entry_broadcast = ttk.Entry(self.form_frame, width=30)
        self.entry_broadcast.grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Label(self.form_frame, text="Broadcast period in secs:").grid(
            row=3, column=0, sticky=tk.E, pady=2
        )
        self.entry_period = ttk.Entry(self.form_frame, width=15)
        self.entry_period.grid(row=3, column=1, sticky=tk.W, pady=2)

        ttk.Label(self.form_frame, text="Custom LuxCore properties:").grid(
            row=4, column=0, sticky=tk.E, pady=2
        )
        self.text_props = tk.Text(
            self.form_frame, height=4, width=40, wrap=tk.NONE
        )
        self.text_props.grid(row=4, column=1, sticky=tk.W, pady=2)

        # Reset button
        self.button_reset = ttk.Button(
            self.form_frame,
            text="Reset configuration",
            command=self.clickedResetConfig,
        )
        self.button_reset.grid(row=5, column=0, columnspan=2, pady=5)

        # Note label
        self.label_note = ttk.Label(
            self.frame_node_config,
            text=(
                'Note: just press "Start node" button '
                "to use the default configuration"
            ),
        )
        self.label_note.pack(fill=tk.X, pady=5)

        # Start/Stop buttons
        self.button_frame = ttk.Frame(self.central_frame)
        self.button_frame.pack(fill=tk.X, pady=5)

        self.button_start = ttk.Button(
            self.button_frame, text="Start node", command=self.clickedStartNode
        )
        self.button_start.pack(side=tk.LEFT, padx=10)

        self.button_stop = ttk.Button(
            self.button_frame,
            text="Stop node",
            command=self.clickedStopNode,
            state=tk.DISABLED,
        )
        self.button_stop.pack(side=tk.RIGHT, padx=10)

        # Log text
        self.text_log = tk.Text(
            self.central_frame, wrap=tk.NONE, state=tk.DISABLED
        )
        self.text_log.pack(fill=tk.BOTH, expand=True)

        # Menu bar
        self.menubar = tk.Menu(self.root)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(
            label="&Quit", command=self.clickedQuit, accelerator="Ctrl+Q"
        )
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.root.config(menu=self.menubar)

        # Keyboard shortcut for Quit
        self.root.bind("<Control-q>", lambda e: self.clickedQuit())

    def clickedQuit(self):
        self.root.destroy()

    def clickedResetConfig(self):
        # Implement your reset logic here
        pass

    def clickedStartNode(self):
        # Implement your start logic here
        self.button_start.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)

    def clickedStopNode(self):
        # Implement your stop logic here
        self.button_start.config(state=tk.NORMAL)
        self.button_stop.config(state=tk.DISABLED)

    def log_message(self, message):
        self.text_log.config(state=tk.NORMAL)
        self.text_log.insert(tk.END, message + "\n")
        self.text_log.see(tk.END)
        self.text_log.config(state=tk.DISABLED)
