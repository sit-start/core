#!/usr/bin/env bash

### Install utilities for Amazon Linux 2023 ###

readonly USER=ec2-user # for Amazon Linux 2023-4
readonly USER_HOME=$(eval echo ~$USER)
readonly MAIN_VENV_PATH="$USER_HOME/.virtualenvs/main"
readonly ARCH=$(uname -m)
if [ "$ARCH" == "aarch64" ]; then
  readonly PYTHON_VER=python3.11
else
  readonly PYTHON_VER=python3.10
fi
readonly CUDA_HOME=/usr/local/cuda
readonly PYTHON_PACKAGES="ipykernel ipympl matplotlib jupyterthemes mplcursors h5py scipy tensorboard grpcio-tools torch-tb-profiler imageio imageio-ffmpeg torch-tb-profiler hydra-core jupyter jupyterlab_widgets Pillow pandas numpy urllib3 ffmpeg scikit-learn tqdm boto3 regex pytest determined typing-extensions sympy filelock fsspec networkx pyyaml sshconf cloudpathlib pigar ray[default,data,train,tune,client] wandb"

function install_core_packages() {
  echo "Installing core packages"
  # Update all system packages to their latest versions
  yum -y update

  # Install development tools first
  yum -y groupinstall "Development tools"

  # Install any remaining tools
  yum -y install emacs cmake cmake3 ninja-build protobuf amazon-efs-utils clang clang-tools-extra amazon-cloudwatch-agent htop
}

function install_yadm() {
  curl -fLo /usr/local/bin/yadm https://github.com/TheLocehiliosan/yadm/raw/master/yadm && chmod a+x /usr/local/bin/yadm
}

function install_awscli() {
  yum -y remove awscli
  mkdir awscli
  pushd awscli
  curl "https://awscli.amazonaws.com/awscli-exe-linux-$ARCH.zip" -o "awscliv2.zip"
  unzip awscliv2.zip
  ./aws/install
  hash aws
  popd
}

function install_gflags_from_source() {
  echo "Installing gflags"
  version="2.2.2"
  mkdir gflags
  pushd gflags
  wget -nv https://github.com/gflags/gflags/archive/refs/tags/v$version.tar.gz
  tar -xzf v$version.tar.gz
  cd gflags-$version
  mkdir build && cd build
  cmake3 .. -DBUILD_SHARED_LIBS=ON
  make install
  popd
}

function install_glog_from_source() {
  echo "Installing glog"
  version="0.6.0"
  mkdir glog
  pushd glog
  wget -nv https://github.com/google/glog/archive/refs/tags/v$version.tar.gz
  tar -xzf v$version.tar.gz
  cd glog-$version
  mkdir build && cd build
  cmake3 .. -DBUILD_SHARED_LIBS=ON
  make install
  popd
}

function install_ffmpeg() {
  # https://www.johnvansickle.com/ffmpeg/faq/
  echo "Installing ffmpeg"
  if [ "$ARCH" == "aarch64" ]; then
    arch="arm64"
  else # x86_64
    arch="amd64"
  fi

  mkdir ffmpeg
  pushd ffmpeg

  build="ffmpeg-6.0.1-$arch-static"
  wget -nv "https://www.johnvansickle.com/ffmpeg/old-releases/$build.tar.xz"
  tar -xvf "$build.tar.xz"
  mv "$build" /usr/local/bin/ffmpeg
  ln -s /usr/local/bin/ffmpeg/ffmpeg /usr/bin/ffmpeg
  ln -s /usr/local/bin/ffmpeg/ffprobe /usr/bin/ffprobe

  popd
}

function install_python() {
  echo "Installing Python and Python packages"
  # Install python; -devel is necessary for some package installations, e.g., psutil
  yum -y install python3-devel "$PYTHON_VER" "$PYTHON_VER-devel" "$PYTHON_VER-pip"
}

function install_python_packages() {
  # Create and activate a default Python virtual environment
  "$PYTHON_VER" -m venv "$MAIN_VENV_PATH"
  source "$MAIN_VENV_PATH/bin/activate"

  # Install Python packages
  pip3 install --upgrade pip
  pip3 install $PYTHON_PACKAGES
}

function install_docker() {
  echo "Installing Docker and the NVIDIA container toolkit"
  # Install/setup docker; requires a reboot for group settings to take
  # effect and for docker to start
  yum -y install docker
  usermod -aG docker $USER
  systemctl enable docker.service
  systemctl enable containerd.service

  # NVIDIA container toolkit. Used by determined-ai
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo |
    tee /etc/yum.repos.d/nvidia-container-toolkit.repo
  yum -y install nvidia-container-toolkit
}

function install_nvidia() {
  echo "Installing NVIDIA driver, CUDA, and cuDNN"
  # based on this:
  # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html
  # NOTE: this fails on amazon linux 2023, but it works on
  # ami-08a800e4b5aa90bb8, which is based on amazon linux 2023. so we
  # can continue to use the new AMI and just update the drivers here on
  # top of what's already installed in the AMI.
  wget -nv "https://us.download.nvidia.com/tesla/535.129.03/NVIDIA-Linux-${ARCH}-535.129.03.run"
  sh "NVIDIA-Linux-${ARCH}-535.129.03.run" --silent --disable-nouveau --tmpdir="$USER_HOME/tmp"

  # CUDA
  suffix=
  if [ "$ARCH" == "aarch64" ]; then
    suffix="_sbsa"
  fi
  cuda_installer="cuda_12.3.2_545.23.08_linux$suffix.run"
  wget -nv "https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/$cuda_installer"
  sh "$cuda_installer" --silent --override \
    --toolkit --samples --toolkitpath=/usr/local/cuda-12.3 \
    --samplespath="$CUDA_HOME" --no-opengl-libs --tmpdir="$USER_HOME/tmp"

  # cuDNN
  if [ "$ARCH" == "aarch64" ]; then
    arch="sbsa"
  else
    arch="x86_64"
  fi
  archive="cudnn-linux-$arch-8.9.7.29_cuda12-archive"
  wget -nv "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-$arch/$archive.tar.xz"
  tar -xf "$archive.tar.xz"
  cp -P "$archive/include/*" "$CUDA_HOME/include/"
  cp -P "$archive/lib/*" "$CUDA_HOME/lib64/"
  chmod a+r "$CUDA_HOME/lib64/*"
}

function install_pytorch() {
  echo "Installing Pytorch"
  source "$MAIN_VENV_PATH/bin/activate"
  pip3 install torch torchvision torchaudio
}

function install_pytorch_from_source() {
  echo "Installing Pytorch from source"
  # Download and install ccache for faster compilation
  wget -nv https://github.com/ccache/ccache/releases/download/v4.8.3/ccache-4.8.3.tar.xz
  tar -xf ccache-4.8.3.tar.xz
  pushd ccache-4.8.3
  cmake .
  make -j $CPUS
  make install
  popd

  # Install release from source
  # NOTE: this may OOM with < 16GB RAM; this can be mitigated by setting
  #       CMAKE_BUILD_PARALLEL_LEVEL to a lower value
  git clone --recursive https://github.com/pytorch/pytorch.git
  pushd pytorch
  # git checkout v2.2.0 # TODO: use a release, not main

  source "$MAIN_VENV_PATH/bin/activate"
  LD_LIBRARY_PATH="$CUDA_HOME/lib64/:$LD_LIBRARY_PATH"
  PATH="$CUDA_HOME/bin:$PATH"
  CMAKE_CUDA_COMPILER=$(which nvcc)

  pip3 install -r requirements.txt
  python3 setup.py install

  popd

  # Refresh the dynamic linker run-time bindings
  ldconfig
}

function install_torchvision_from_source() {
  source "$MAIN_VENV_PATH/bin/activate"
  LD_LIBRARY_PATH="$CUDA_HOME/lib64/:$LD_LIBRARY_PATH"
  PATH="$CUDA_HOME/bin:$PATH"
  CMAKE_CUDA_COMPILER=$(which nvcc)

  echo "Installing libjpeg-turbo and libpng"
  yum -y install libjpeg-turbo-devel libpng-devel

  echo "Installing nvJPEG2000"
  wget https://developer.download.nvidia.com/compute/nvjpeg2000/0.7.5/local_installers/nvjpeg2000-local-repo-rhel9-0.7.5-1.0-1.${ARCH}.rpm
  rpm -i nvjpeg2000-local-repo-rhel9-0.7.5-1.0-1.aarch64.rpm
  dnf clean all
  dnf -y install nvjpeg2k

  echo "Installing torchvision from source"
  wget https://github.com/pytorch/vision/archive/refs/tags/v0.17.0.tar.gz
  tar -xf v0.17.0.tar.gz
  pushd vision-0.17.0
  python3 setup.py install
  popd
}

function install_python_cleanup() {
  chown -R $USER $MAIN_VENV_PATH
}

function install() {
  set -v
  # use the larger root volume for temp files during script execution
  mkdir -p "$USER_HOME/tmp"
  pushd "$USER_HOME/tmp"

  for req in "${@}"; do
    "install_$req"
  done

  popd
  set +v
}

function install_g5g() {
  install core_packages yadm gflags_from_source glog_from_source ffmpeg python \
    python_packages nvidia docker pytorch_from_source torchvision_from_source \
    python_cleanup
}

function install_g5() {
  install core_packages yadm awscli gflags_from_source glog_from_source ffmpeg \
    python_packages pytorch python_cleanup
}
