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

import logging
import tkinter as tk

import pyluxcore

from .mainwindow import MainWindow
from .._renderfarm import DEFAULT_PORT, RenderFarmNode
from .._utils import loghandler

logger = logging.getLogger(loghandler.loggerName + ".luxcorenetnodeui")


class MainApp(MainWindow):
    def __init__(self, root):
        super().__init__(root)
        self.__ResetConfigUI()
        self.center_window()
        logger.info("LuxCore %s", pyluxcore.Version())
        self.renderFarmNode = None
        self.log_message("Waiting for configuration...")

    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def __ResetConfigUI(self):
        self.entry_ip.delete(0, tk.END)
        self.entry_port.delete(0, tk.END)
        self.entry_port.insert(0, str(DEFAULT_PORT))
        self.entry_broadcast.delete(0, tk.END)
        self.entry_broadcast.insert(0, "<broadcast>")
        self.entry_period.delete(0, tk.END)
        self.entry_period.insert(0, str(3.0))
        self.text_props.delete("1.0", tk.END)

    def clickedResetConfig(self):
        self.__ResetConfigUI()

    def clickedStartNode(self):
        self.frame_node_config.pack_forget()
        self.button_start.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)

        self.renderFarmNode = RenderFarmNode(
            self.entry_ip.get(),
            int(self.entry_port.get()),
            self.entry_broadcast.get(),
            float(self.entry_period.get()),
            pyluxcore.Properties(),
        )
        self.renderFarmNode.Start()
        self.log_message("Started")

    def clickedStopNode(self):
        self.renderFarmNode.Stop()
        self.renderFarmNode = None

        self.button_start.config(state=tk.NORMAL)
        self.button_stop.config(state=tk.DISABLED)
        self.frame_node_config.pack(fill=tk.X, pady=5)
        self.log_message("Waiting for configuration...")

    def clickedQuit(self):
        self.root.destroy()

    def closeEvent(self):
        if self.renderFarmNode:
            self.renderFarmNode.Stop()
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


if __name__ == "__main__":
    main()
