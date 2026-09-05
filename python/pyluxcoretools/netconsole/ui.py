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

import time
import logging
import socket
import tkinter as tk
from tkinter import messagebox, filedialog
import functools

import pyluxcore
from .._renderfarm import RenderFarm, NodeDiscoveryType
from .._renderfarm import jobsingleimage
from .._utils import loghandler, netbeacon
from .mainwindow import MainWindow
from .addnodedialog import AddNodeDialog

logger = logging.getLogger(loghandler.loggerName + ".luxcorenetconsoleui")


def __format_samples_sec(val):
    return (
        f"{val / 1000.0:.1f}K"
        if val < 1000000.0
        else f"{val / 1000000.0:.1f}M"
    )


def _validate_int(value):
    return value.isdigit() or value == ""


def _validate_positive_int(value):
    return value.isdigit() and int(value) > 0 or value == ""


class MainApp(MainWindow):
    def __init__(self, root):
        super().__init__(root)
        self.center_window()
        self.root.protocol("WM_DELETE_WINDOW", self.clicked_quit)

        # Initialize render farm
        self.render_farm = RenderFarm()
        self.render_farm.Start()

        # Start the beacon receiver
        self.beacon = netbeacon.NetBeaconReceiver(
            functools.partial(self.__node_discovery_callback, self)
        )
        self.beacon.Start()

        # Set up callbacks
        self.render_farm.SetJobsUpdateCallBack(
            functools.partial(self.__render_farm_jobs_update_callback, self)
        )
        self.render_farm.SetNodesUpdateCallBack(
            functools.partial(self.__render_farm_nodes_update_callback, self)
        )

        # Disable current job tab initially
        self.notebook.tab(0, state="disabled")
        self.notebook.select(1)

        # Set validators for entries
        self.entry_halt_spp.config(
            validate="key",
            validatecommand=(self.root.register(_validate_int), "%P"),
        )
        self.entry_halt_time.config(
            validate="key",
            validatecommand=(self.root.register(_validate_int), "%P"),
        )
        self.entry_film_update.config(
            validate="key",
            validatecommand=(self.root.register(_validate_int), "%P"),
        )
        self.entry_stats_period.config(
            validate="key",
            validatecommand=(
                self.root.register(_validate_positive_int),
                "%P",
            ),
        )

        logger.info("LuxCore %s", pyluxcore.Version())

        # Initial updates
        self.__render_farm_jobs_update_callback()
        self.__render_farm_nodes_update_callback()

    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def __node_discovery_callback(self, ip_address, port):
        self.render_farm.DiscoveredNode(
            ip_address, port, NodeDiscoveryType.AUTO_DISCOVERED
        )

    def __render_farm_jobs_update_callback(self):
        self.__update_current_job_tab()
        self.__update_queued_jobs_tab()

    def __render_farm_nodes_update_callback(self):
        self.__update_nodes_tab()

    def __update_nodes_tab(self):
        self.nodes_list.delete(0, tk.END)
        for idx, node in enumerate(self.render_farm.GetNodesList()):
            self.nodes_list.insert(
                tk.END, f"{idx}: {node.GetKey()} - {node.state.name}"
            )

    def __update_current_job_tab(self):
        if current_job := self.render_farm.currentJob:
            self.notebook.tab(0, state="normal")
            self.label_render_cfg.config(
                text=current_job.GetRenderConfigFileName()
            )
            self.label_film_file.config(text=current_job.GetFilmFileName())
            self.label_image_file.config(text=current_job.GetImageFileName())
            self.label_work_dir.config(text=current_job.GetWorkDirectory())

            rendering_start_time = current_job.GetStartTime()
            self.label_start_time.config(
                text=time.strftime(
                    "%H:%M:%S %Y/%m/%d", time.localtime(rendering_start_time)
                )
            )

            dt = time.time() - rendering_start_time
            self.label_rendering_time.config(
                text=time.strftime("%H:%M:%S", time.gmtime(dt))
            )

            self.label_samples_pixel.config(
                text=f"{current_job.GetSamplesPixel():.1f}"
            )
            self.label_samples_sec.config(
                text=__format_samples_sec(current_job.GetSamplesSec())
            )

            self.entry_halt_spp.delete(0, tk.END)
            self.entry_halt_spp.insert(0, str(current_job.GetFilmHaltSPP()))

            self.entry_halt_time.delete(0, tk.END)
            self.entry_halt_time.insert(0, str(current_job.GetFilmHaltTime()))

            self.entry_film_update.delete(0, tk.END)
            self.entry_film_update.insert(
                0, str(current_job.GetFilmUpdatePeriod())
            )

            self.entry_stats_period.delete(0, tk.END)
            self.entry_stats_period.insert(
                0, str(current_job.GetStatsPeriod())
            )

            self.__update_current_rendering_image()
        else:
            self.notebook.tab(0, state="disabled")

    def __update_queued_jobs_tab(self):
        self.queued_jobs_list.delete(0, tk.END)
        for idx, job in enumerate(self.render_farm.GetQueuedJobList()):
            self.queued_jobs_list.insert(
                tk.END, f"{idx}: {job.GetRenderConfigFileName()}"
            )

    def __update_current_rendering_image(self):
        if current_job := self.render_farm.currentJob:
            try:
                from PIL import Image, ImageTk

                image = Image.open(current_job.GetImageFileName())
                photo = ImageTk.PhotoImage(image)
                self.label_rendering_image.config(image=photo)
                self.label_rendering_image.image = photo
            except Exception:
                self.label_rendering_image.config(
                    image=None, text="Waiting for film download and merge"
                )
        else:
            self.label_rendering_image.config(image=None, text="N/A")

    def clicked_add_node(self):
        dialog = AddNodeDialog(self.root)
        if result := dialog.show():
            ip_address = result["ip"]
            port = int(result["port"])
            try:
                socket.inet_aton(ip_address)
            except socket.error:
                messagebox.showerror(
                    "Error", f"Invalid IP address: {ip_address}"
                )
                return
            self.render_farm.DiscoveredNode(
                ip_address, port, NodeDiscoveryType.MANUALLY_DISCOVERED
            )

    def clicked_add_job(self):
        file_to_render = filedialog.askopenfilename(
            title="Open file to render",
            filetypes=[("Binary render configuration", "*.bcf")],
        )
        if file_to_render:
            logger.info(
                "Creating single image render farm job: %s", file_to_render
            )
            render_farm_job = jobsingleimage.RenderFarmJobSingleImage(
                self.render_farm, file_to_render
            )
            self.render_farm.AddJob(render_farm_job)
            self.__update_current_job_tab()
            self.__update_queued_jobs_tab()
            self.notebook.select(0)

    def clicked_remove_pending_jobs(self):
        self.render_farm.RemovePendingJobs()
        self.__update_queued_jobs_tab()

    def clicked_force_film_merge(self):
        if current_job := self.render_farm.currentJob:
            current_job.ForceFilmMerge()

    def clicked_force_film_download(self):
        if current_job := self.render_farm.currentJob:
            current_job.ForceFilmDownload()

    def clicked_finish_job(self):
        self.render_farm.StopCurrentJob()
        self.__update_current_job_tab()
        self.__update_queued_jobs_tab()

    def clicked_refresh_nodes_list(self):
        self.__render_farm_nodes_update_callback()

    def clicked_quit(self):
        self.beacon.Stop()
        self.render_farm.Stop()
        self.root.destroy()


def ui():
    try:
        pyluxcore.Init(loghandler.LuxCoreLogHandler)
        root = tk.Tk()
        app = MainApp(root)
        app.mainloop()
    finally:
        pyluxcore.SetLogHandler(None)


def main():
    ui()
