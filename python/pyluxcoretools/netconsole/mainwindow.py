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
        self.root.title("PyLuxCore Tool NetConsole")
        self.root.geometry("800x800")

        # Central widget
        self.central_frame = ttk.Frame(self.root, padding="10")
        self.central_frame.pack(fill=tk.BOTH, expand=True)

        # Notebook (Tab Widget)
        self.notebook = ttk.Notebook(self.central_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Current Job Tab
        self.current_job_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.current_job_tab, text="Current job")

        self.setup_current_job_tab()

        # Queued Jobs Tab
        self.queued_jobs_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.queued_jobs_tab, text="Queued jobs")

        self.setup_queued_jobs_tab()

        # Nodes Tab
        self.nodes_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.nodes_tab, text="Nodes")

        self.setup_nodes_tab()

        # Log Text
        self.log_text = tk.Text(
            self.central_frame, wrap=tk.NONE, state=tk.DISABLED, height=10
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Menu Bar
        self.menu_bar = tk.Menu(self.root)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(
            label="&Quit", command=self.clicked_quit, accelerator="Ctrl+Q"
        )
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.root.config(menu=self.menu_bar)

        # Keyboard shortcut for Quit
        self.root.bind("<Control-q>", lambda e: self.clicked_quit())

    def setup_current_job_tab(self):
        # Scrollable Frame
        self.current_job_scroll = ttk.Frame(self.current_job_tab)
        self.current_job_scroll.pack(fill=tk.BOTH, expand=True)

        # Information Group
        self.info_group = ttk.LabelFrame(
            self.current_job_scroll, text="Information", padding="10"
        )
        self.info_group.pack(fill=tk.X, pady=5)

        # Information Labels
        ttk.Label(
            self.info_group, text="Render configuration file name:"
        ).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_render_cfg = ttk.Label(self.info_group, text="")
        self.label_render_cfg.grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2
        )

        ttk.Label(self.info_group, text="Film file name:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.label_film_file = ttk.Label(self.info_group, text="")
        self.label_film_file.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(self.info_group, text="Image file name:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.label_image_file = ttk.Label(self.info_group, text="")
        self.label_image_file.grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=2
        )

        ttk.Label(self.info_group, text="Work directory:").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.label_work_dir = ttk.Label(self.info_group, text="")
        self.label_work_dir.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(self.info_group, text="Start time:").grid(
            row=4, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.label_start_time = ttk.Label(self.info_group, text="")
        self.label_start_time.grid(
            row=4, column=1, sticky=tk.W, padx=5, pady=2
        )

        ttk.Label(self.info_group, text="Rendering time:").grid(
            row=5, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.label_rendering_time = ttk.Label(self.info_group, text="")
        self.label_rendering_time.grid(
            row=5, column=1, sticky=tk.W, padx=5, pady=2
        )

        ttk.Label(self.info_group, text="Samples/pixel:").grid(
            row=6, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.label_samples_pixel = ttk.Label(self.info_group, text="")
        self.label_samples_pixel.grid(
            row=6, column=1, sticky=tk.W, padx=5, pady=2
        )

        ttk.Label(self.info_group, text="Samples/sec:").grid(
            row=7, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.label_samples_sec = ttk.Label(self.info_group, text="")
        self.label_samples_sec.grid(
            row=7, column=1, sticky=tk.W, padx=5, pady=2
        )

        # Configuration Group
        self.config_group = ttk.LabelFrame(
            self.current_job_scroll, text="Configuration", padding="10"
        )
        self.config_group.pack(fill=tk.X, pady=5)

        ttk.Label(
            self.config_group, text="Halt at samples/pixel (0 disabled):"
        ).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.entry_halt_spp = ttk.Entry(self.config_group, width=10)
        self.entry_halt_spp.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(
            self.config_group, text="Halt at time (in secs, 0 disabled):"
        ).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.entry_halt_time = ttk.Entry(self.config_group, width=10)
        self.entry_halt_time.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(
            self.config_group, text="Film update period (in secs):"
        ).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.entry_film_update = ttk.Entry(self.config_group, width=10)
        self.entry_film_update.grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=2
        )

        ttk.Label(
            self.config_group, text="Statistics print period (in secs):"
        ).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.entry_stats_period = ttk.Entry(self.config_group, width=10)
        self.entry_stats_period.grid(
            row=3, column=1, sticky=tk.W, padx=5, pady=2
        )

        # Commands Group
        self.commands_group = ttk.LabelFrame(
            self.current_job_scroll, text="Commands", padding="10"
        )
        self.commands_group.pack(fill=tk.X, pady=5)

        self.button_force_film_merge = ttk.Button(
            self.commands_group,
            text="Force film merge",
            command=self.clicked_force_film_merge,
        )
        self.button_force_film_merge.grid(row=0, column=1, padx=5, pady=5)

        self.button_force_film_download = ttk.Button(
            self.commands_group,
            text="Force film download",
            command=self.clicked_force_film_download,
        )
        self.button_force_film_download.grid(row=0, column=2, padx=5, pady=5)

        self.button_finish_job = ttk.Button(
            self.commands_group,
            text="Finish job",
            command=self.clicked_finish_job,
        )
        self.button_finish_job.grid(row=0, column=3, padx=5, pady=5)

        # Rendering Group
        self.rendering_group = ttk.LabelFrame(
            self.current_job_scroll, text="Rendering", padding="10"
        )
        self.rendering_group.pack(fill=tk.X, pady=5)

        self.label_rendering_image = ttk.Label(
            self.rendering_group, text="Waiting for film download and merge"
        )
        self.label_rendering_image.pack()

    def setup_queued_jobs_tab(self):
        # Add Job Button
        self.button_add_job = ttk.Button(
            self.queued_jobs_tab, text="Add job", command=self.clicked_add_job
        )
        self.button_add_job.pack(pady=5)

        # Remove Pending Jobs Button
        self.button_remove_pending = ttk.Button(
            self.queued_jobs_tab,
            text="Remove pending jobs",
            command=self.clicked_remove_pending_jobs,
        )
        self.button_remove_pending.pack(pady=5)

        # Queued Jobs Scrollable List
        self.queued_jobs_scroll = ttk.Frame(self.queued_jobs_tab)
        self.queued_jobs_scroll.pack(fill=tk.BOTH, expand=True)

        self.queued_jobs_list = tk.Listbox(self.queued_jobs_scroll)
        self.queued_jobs_list.pack(fill=tk.BOTH, expand=True)

    def setup_nodes_tab(self):
        # Add Node Button
        self.button_add_node = ttk.Button(
            self.nodes_tab, text="Add node", command=self.clicked_add_node
        )
        self.button_add_node.pack(pady=5)

        # Refresh Nodes List Button
        self.button_refresh_nodes = ttk.Button(
            self.nodes_tab,
            text="Refresh nodes list",
            command=self.clicked_refresh_nodes_list,
        )
        self.button_refresh_nodes.pack(pady=5)

        # Nodes Scrollable List
        self.nodes_scroll = ttk.Frame(self.nodes_tab)
        self.nodes_scroll.pack(fill=tk.BOTH, expand=True)

        self.nodes_list = tk.Listbox(self.nodes_scroll)
        self.nodes_list.pack(fill=tk.BOTH, expand=True)

    def log_message(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def clicked_quit(self):
        self.root.destroy()

    def clicked_force_film_merge(self):
        self.log_message("Force film merge clicked")

    def clicked_force_film_download(self):
        self.log_message("Force film download clicked")

    def clicked_finish_job(self):
        self.log_message("Finish job clicked")

    def clicked_add_job(self):
        self.log_message("Add job clicked")

    def clicked_remove_pending_jobs(self):
        self.log_message("Remove pending jobs clicked")

    def clicked_add_node(self):
        self.log_message("Add node clicked")

    def clicked_refresh_nodes_list(self):
        self.log_message("Refresh nodes list clicked")
