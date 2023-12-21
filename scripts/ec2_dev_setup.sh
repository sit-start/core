#!/usr/bin/env bash

readonly USER_HOME=/home/ec2-user
readonly MAIN_VENV_PATH="${USER_HOME}/.virtualenvs/main"
readonly ARCH=arm64
readonly PYTHON_VER=python3.11

function main() {
  # Install development tools first
  yum -y groupinstall "Development tools"

  # Install any remaining native tools
  yum -y install emacs

  # Install Python 3.11; -devel is necessary for some package installations, e.g., psutil
  yum -y install "${PYTHON_VER}" "${PYTHON_VER}-devel" "${PYTHON_VER}-pip" python3-devel

  # Create and activate a default Python virtual environment
  "${PYTHON_VER}" -m venv "${MAIN_VENV_PATH}"
  source "${MAIN_VENV_PATH}"/bin/activate

  # Install Python packages
  pip3 install --upgrade pip
  pip3 install ipykernel torch torchvision ipympl matplotlib jupyterthemes mplcursors h5py scipy tensorboard grpcio-tools torch-tb-profiler imageio imageio-ffmpeg torch-tb-profiler hydra-core jupyter jupyterlab_widgets Pillow pandas numpy urllib3 ffmpeg scikit-learn tqdm boto3

  # Install yadm
  curl -fLo /usr/local/bin/yadm https://github.com/TheLocehiliosan/yadm/raw/master/yadm && chmod a+x /usr/local/bin/yadm
}

main "${@}"
