import ipympl  # noqa: F401
from IPython.core.getipython import get_ipython
from IPython.display import HTML, display

# ensure figure background/foreground colors are consistent with the editor
_color_consistency_styling = HTML(
    """<style>
  .cell-output-ipywidget-background {
      background-color: transparent !important;
  }
  :root {
      --jp-widgets-color: var(--vscode-editor-foreground);
      --jp-widgets-font-size: var(--vscode-editor-font-size);
  }"""
)


# setup interactive plots, equivalent to %matplotlib widget; requires ipympl
def setup_interactive_plots():
    ipython = get_ipython()
    if ipython:
        # requires ipympl
        ipython.run_line_magic("matplotlib", "widget")
    else:
        raise RuntimeError("iPython not found")
    display(_color_consistency_styling)


def figure():
    import matplotlib.pyplot as plt

    existing_fignums = plt.get_fignums()
    fig_num = 1 if len(existing_fignums) == 0 else max(existing_fignums) + 1
    fig = plt.figure(num=fig_num, clear=True)
    fig_num += 1
    return fig
