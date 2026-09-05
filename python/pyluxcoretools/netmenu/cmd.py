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

from .menuwindow_tk import setup_ui
from ..netnode import ui as netnode_ui
from ..netconsole import ui as netconsole_ui
from .._utils import loghandler


logger = logging.getLogger(loghandler.loggerName + ".luxcoremenu")


class MenuApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.selectedTool = "None"
        self.title("LuxCoreMenu")
        self.geometry("300x200")
        self.center_window()

        # Assuming menuwindow.setup_ui(self) sets up the UI
        setup_ui(self)

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def clickedNetNode(self):
        self.selectedTool = "NetNode"
        self.destroy()

    def clickedNetConsole(self):
        self.selectedTool = "NetConsole"
        self.destroy()


def main():
    app = MenuApp()
    app.mainloop()

    if app.selectedTool == "NetNode":
        netnode_ui()
    elif app.selectedTool == "NetConsole":
        netconsole_ui()
