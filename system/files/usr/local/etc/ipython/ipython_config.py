c = get_config()  # type: ignore # noqa

# @source: https://stackoverflow.com/questions/5364050/reloading-submodules-in-ipython
c.InteractiveShellApp.extensions = ["autoreload"]
c.InteractiveShellApp.exec_lines = ["%autoreload 2"]
