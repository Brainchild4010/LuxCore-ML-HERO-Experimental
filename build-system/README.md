## Helpers and other stuff for Build process

Root contents:
- `build-settings.json`: static settings for build process, including version of the dependencies. Used by CMake, Github Workflows and others.
- `ccache.conf`: configuration for ccache. Used by Github Workflows

Folder contents:
- `cmake`: additional modules for CMake. Used by CMakeLists.txt.
- `conan`: project's conan recipe. Used by `make` helpers (see below)
- `debug`: (handcrafted) scripts for build debugging. Run on command line
- `luxmake`: underlying Python scripts of `make` wrapper - enabling `make deps`, `make <target>`, `make clean` etc. Called by `GNUMakefile` and `make.bat`
- `wheels`: helpers for wheel building. Called by Github Workflows.
