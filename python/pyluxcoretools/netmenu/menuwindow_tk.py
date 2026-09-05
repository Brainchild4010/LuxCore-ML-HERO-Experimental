import tkinter as tk
from tkinter import ttk


def setup_ui(root):
    # Main window setup
    root.title("PyLuxCore Tools Menu")
    root.geometry("260x240")

    # Central frame
    central_frame = ttk.Frame(root, padding="10")
    central_frame.pack(fill=tk.BOTH, expand=True)

    # Vertical layout
    main_layout = ttk.Frame(central_frame)
    main_layout.pack(fill=tk.BOTH, expand=True)

    # NetNode button
    netnode_button = ttk.Button(
        main_layout, text="NetNode", command=root.clickedNetNode
    )
    netnode_button.pack(pady=5, fill=tk.X)

    # NetConsole button
    netconsole_button = ttk.Button(
        main_layout, text="NetConsole", command=root.clickedNetConsole
    )
    netconsole_button.pack(pady=5, fill=tk.X)

    # Spacer
    spacer = ttk.Frame(main_layout, height=40)
    spacer.pack(fill=tk.X, pady=5)

    # Quit button
    quit_button = ttk.Button(main_layout, text="Quit", command=root.destroy)
    quit_button.pack(pady=5, fill=tk.X)

    # Menu bar
    menubar = tk.Menu(root)
    tools_menu = tk.Menu(menubar, tearoff=0)
    tools_menu.add_command(
        label="&Quit", command=root.destroy, accelerator="Ctrl+Q"
    )
    menubar.add_cascade(label="Tools", menu=tools_menu)
    root.config(menu=menubar)

    # Keyboard shortcut for Quit
    root.bind("<Control-q>", lambda e: root.destroy())
