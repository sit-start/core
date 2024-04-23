#!/usr/bin/env bash

# Activate a venv by path, and failing that, a conda env by name.
# If no path or name is specified, defaults to `.venv`.
function activate_py_env() {
  if [ "$#" -lt 1 ]; then
    env=./.venv
  else
    env="$1"
  fi
  # first, try activating as a virtual environment
  if [ -d "$env" ]; then
    # export the deactivate function for use in scripts/sub-shells
    source "$env/bin/activate" && export -f deactivate
  # otherwise use conda
  else
    source activate "$env" >/dev/null 2>&1
  fi
}

# Deactivate any conda or virtual python environment.
function deactivate_py_env() {
  test -n "${VIRTUAL_ENV}" && deactivate
  test -n "${CONDA_DEFAULT_ENV}" && conda deactivate
}

# Activate the default python environment. This is a venv called 'main' on
# most machines, but it's a pre-installed conda 'pytorch' env on some configs.
export DEFAULT_VENV="${HOME}/.virtualenvs/main"
function activate_default_py_env() {
  activate_py_env "$DEFAULT_VENV" || activate_py_env pytorch
}

# Reactivate the active python environment.
function reactivate_py_env() {
  test -n "${VIRTUAL_ENV}" && activate_py_env "${VIRTUAL_ENV}"
  test -n "${CONDA_DEFAULT_ENV}" && activate_py_env "${CONDA_DEFAULT_ENV}"
}
