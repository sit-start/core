from pathlib import Path
from pkgutil import extend_path

# @source: https://stackoverflow.com/questions/26058978/python-import-different-subpackages-with-the-same-root-packge-name-and-different
__path__ = extend_path(__path__, __name__)

REPO_ROOT = str(Path(__file__).parents[2])
PYTHON_ROOT = f"{REPO_ROOT}/python"
