#!/usr/bin/env bash

# usage: join_by sep tok1 tok2 ... tokn
function join_by {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

### Install utilities for Amazon Linux 2023 ###
readonly USER=ec2-user # for Amazon Linux 2023-4
USER_HOME=$(eval echo ~$USER)
readonly USER_HOME
readonly MAIN_VENV_PATH="$USER_HOME/.virtualenvs/main"
ARCH=$(uname -m)
readonly ARCH
if [ "$ARCH" == "aarch64" ]; then
  readonly PYTHON_VER=python3.11
else
  readonly PYTHON_VER=python3.10
fi
readonly CUDA_HOME=/usr/local/cuda
readonly PYTHON_PACKAGES=(
  ipykernel ipympl matplotlib jupyterthemes mplcursors h5py scipy tensorboard
  grpcio-tools torch-tb-profiler imageio imageio-ffmpeg torch-tb-profiler
  hydra-core jupyter jupyterlab_widgets Pillow pandas numpy urllib3 ffmpeg
  scikit-learn tqdm boto3 regex pytest determined typing-extensions sympy
  filelock fsspec networkx pyyaml sshconf cloudpathlib pigar
  "ray[default,data,train,tune,client]" wandb pytorch-lightning
)
readonly YUM_PACKAGES=(
  emacs cmake cmake3 ninja-build protobuf amazon-efs-utils clang
  clang-tools-extra amazon-cloudwatch-agent htop
)

function install_core_packages() {
  # Update all system packages to their latest versions
  yum -y update
  # Install development tools first
  yum -y groupinstall "Development tools"
  # Install any remaining tools
  yum -y install "${YUM_PACKAGES[@]}"
}

function install_github() {
  if [ "$ARCH" == "aarch64" ]; then
    arch="arm64"
  else # x86_64
    arch="amd64"
  fi
  version=2.45.0

  url=$(join_by '' \
    "https://github.com/cli/cli/releases/download/v${version}/" \
    "gh_${version}_linux_${arch}.rpm")
  wget "$url"
  rpm -i "gh_${version}_linux_${arch}.rpm"
}

function install_yadm() {
  url=" https://github.com/TheLocehiliosan/yadm/raw/master/yadm"
  curl -fLo /usr/local/bin/yadm "$url"
  chmod a+x /usr/local/bin/yadm
}

function install_grafana() {
  url="https://dl.grafana.com/oss/release/grafana-8.3.3-1.$ARCH.rpm"
  yum install -y "$url"
}

function install_fluent_bit() {
  glibc_ver_prefix="ldd (GNU libc)"
  glibc_ver=$(ldd --version |
    grep "$glibc_ver_prefix" | sed "s/$glibc_ver_prefix \(.*\)$/\1/g")

  if [ "$glibc_ver" == "2.26" ]; then
    base_url="https://packages.fluentbit.io/amazonlinux/2/"
  elif [ "$glibc_ver" == "2.34" ]; then
    base_url="https://packages.fluentbit.io/amazonlinux/2023/"
  else
    echo "Unrecognized platform with glibc version $glibc_ver"
    return
  fi

  echo "[fluent-bit]
name = Fluent Bit
baseurl = $base_url
gpgcheck=1
gpgkey=https://packages.fluentbit.io/fluentbit.key
enabled=1" | tee /etc/yum.repos.d/fluent-bit.repo
  yum -y install fluent-bit
}

function install_loki() {
  # https://rpm.grafana.com/
  echo "[grafana]
name=grafana
baseurl=https://rpm.grafana.com
repo_gpgcheck=1
enabled=1
gpgcheck=1
gpgkey=https://rpm.grafana.com/gpg.key
sslverify=1
sslcacert=/etc/pki/tls/certs/ca-bundle.crt" |
    tee /etc/yum.repos.d/grafana.repo
  yum -y install loki promtail
}

function install_prometheus() {
  # @source: https://devopscube.com/install-configure-prometheus-linux/
  if [ "$ARCH" == "aarch64" ]; then
    arch="arm64"
  else # x86_64
    arch="amd64"
  fi

  version="2.49.1"
  prometheus="prometheus-$version.linux-$arch"
  url=$(join_by '' \
    "https://github.com/prometheus/prometheus/releases/download/" \
    "v$version/$prometheus.tar.gz")
  wget -nv "$url"
  tar -xzf $prometheus.tar.gz

  # Create a Prometheus user, required directories, and make Prometheus the
  # user as the owner of those directories.
  useradd --no-create-home --shell /bin/false prometheus
  mkdir /etc/prometheus
  mkdir /var/lib/prometheus
  chown prometheus:prometheus /etc/prometheus
  chown prometheus:prometheus /var/lib/prometheus

  # Copy prometheus and promtool binary from prometheus-files folder to
  # /usr/local/bin and change the ownership to prometheus user.
  cp "$prometheus/prometheus" /usr/local/bin/
  cp "$prometheus/promtool" /usr/local/bin/
  chown prometheus:prometheus /usr/local/bin/prometheus
  chown prometheus:prometheus /usr/local/bin/promtool

  # Move the consoles and console_libraries directories from prometheus-files
  # to /etc/prometheus folder and change the ownership to prometheus user.
  cp -r $prometheus/consoles /etc/prometheus
  cp -r $prometheus/console_libraries /etc/prometheus
  chown -R prometheus:prometheus /etc/prometheus/consoles
  chown -R prometheus:prometheus /etc/prometheus/console_libraries
}

function install_awscli() {
  yum -y remove awscli
  url="https://awscli.amazonaws.com/awscli-exe-linux-$ARCH.zip"
  curl "$url" -o "awscliv2.zip"
  unzip awscliv2.zip
  ./aws/install
  hash aws
}

function install_gflags_from_source() {
  version="2.2.2"
  url="https://github.com/gflags/gflags/archive/refs/tags/v$version.tar.gz"
  wget -nv "$url"
  tar -xzf v$version.tar.gz
  cd gflags-$version || return 1
  mkdir build
  cd build || {
    echo "Failed to enter build"
    return
  }
  cmake3 .. -DBUILD_SHARED_LIBS=ON
  make install
}

function install_glog_from_source() {
  version="0.6.0"
  url="https://github.com/google/glog/archive/refs/tags/v$version.tar.gz"
  wget -nv "$url"
  tar -xzf v$version.tar.gz
  cd glog-$version || {
    echo "Failed to enter glog-$version"
    return
  }
  mkdir build
  cd build || {
    echo "Failed to enter build"
    return
  }
  cmake3 .. -DBUILD_SHARED_LIBS=ON
  make install
}

function install_ffmpeg() {
  # @source: https://www.johnvansickle.com/ffmpeg/faq/
  if [ "$ARCH" == "aarch64" ]; then
    arch="arm64"
  else # x86_64
    arch="amd64"
  fi

  build="ffmpeg-6.0.1-$arch-static"
  url="https://www.johnvansickle.com/ffmpeg/old-releases/$build.tar.xz"
  wget -nv "$url"
  tar -xvf "$build.tar.xz"
  mv "$build" /usr/local/bin/ffmpeg
  ln -s /usr/local/bin/ffmpeg/ffmpeg /usr/bin/ffmpeg
  ln -s /usr/local/bin/ffmpeg/ffprobe /usr/bin/ffprobe
}

function install_python() {
  # Install python; -devel is necessary for some package installations,
  # e.g., psutil
  targets="python3-devel $PYTHON_VER $PYTHON_VER-devel $PYTHON_VER-pip"
  yum -y install "$targets"
}

function install_python_packages() {
  # Create and activate a default Python virtual environment
  "$PYTHON_VER" -m venv "$MAIN_VENV_PATH"
  source "$MAIN_VENV_PATH/bin/activate"

  # Install Python packages
  pip3 install --upgrade pip
  pip3 install "${PYTHON_PACKAGES[@]}"
}

function install_docker() {
  # Install/setup docker; requires a reboot for group settings to take
  # effect and for docker to start
  yum -y install docker
  usermod -aG docker $USER
  systemctl enable docker.service
  systemctl enable containerd.service

  # NVIDIA container toolkit. Used by determined-ai
  url=$(join_by '' \
    "https://nvidia.github.io/libnvidia-container/stable/rpm/" \
    "nvidia-container-toolkit.repo")
  curl -s -L "$url" | tee /etc/yum.repos.d/nvidia-container-toolkit.repo
  yum -y install nvidia-container-toolkit
}

function install_nvidia() {
  # @source: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html
  # NOTE: this fails on amazon linux 2023, but it works on
  # ami-08a800e4b5aa90bb8, which is based on amazon linux 2023. so we
  # can continue to use the new AMI and just update the drivers here on
  # top of what's already installed in the AMI.
  url=$(join_by '' \
    "https://us.download.nvidia.com/tesla/535.129.03/" \
    "NVIDIA-Linux-${ARCH}-535.129.03.run")
  wget -nv "$url"
  sh "NVIDIA-Linux-${ARCH}-535.129.03.run" \
    --silent --disable-nouveau --tmpdir="$USER_HOME/tmp"

  # CUDA
  suffix=
  if [ "$ARCH" == "aarch64" ]; then
    suffix="_sbsa"
  fi
  cuda_installer="cuda_12.3.2_545.23.08_linux$suffix.run"
  url=$(join_by '' \
    "https://developer.download.nvidia.com/compute/cuda/12.3.2/" \
    "local_installers/$cuda_installer")
  wget -nv "$url"
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
  url=$(join_by '' \
    "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/" \
    "linux-$arch/$archive.tar.xz")
  wget -nv "$url"
  tar -xf "$archive.tar.xz"
  cp -P "$archive/include/*" "$CUDA_HOME/include/"
  cp -P "$archive/lib/*" "$CUDA_HOME/lib64/"
  chmod a+r "$CUDA_HOME/lib64/*"
}

function install_pytorch() {
  source "$MAIN_VENV_PATH/bin/activate"
  pip3 install torch torchvision torchaudio
}

function install_pytorch_from_source() {
  echo "Installing Pytorch from source"
  # Download and install ccache for faster compilation
  url=$(join_by '' \
    "https://github.com/ccache/ccache/releases/download/v4.8.3/" \
    "ccache-4.8.3.tar.xz")
  wget -nv "$url"
  tar -xf ccache-4.8.3.tar.xz
  pushd ccache-4.8.3 || return 1
  cmake .
  make -j "$CPUS"
  make install
  popd || return 1

  # Install release from source
  # NOTE: this may OOM with < 16GB RAM; this can be mitigated by setting
  #       CMAKE_BUILD_PARALLEL_LEVEL to a lower value
  version=v2.2.0
  git clone --depth 1 --branch $version --recursive \
    https://github.com/pytorch/pytorch.git

  source "$MAIN_VENV_PATH/bin/activate"
  LD_LIBRARY_PATH="$CUDA_HOME/lib64/:$LD_LIBRARY_PATH"
  PATH="$CUDA_HOME/bin:$PATH"
  CMAKE_CUDA_COMPILER=$(which nvcc)

  pip3 install -r requirements.txt
  python3 setup.py install

  # Refresh the dynamic linker run-time bindings
  ldconfig
}

function install_torchvision_from_source() {
  source "$MAIN_VENV_PATH/bin/activate"
  LD_LIBRARY_PATH="$CUDA_HOME/lib64/:$LD_LIBRARY_PATH"
  PATH="$CUDA_HOME/bin:$PATH"
  # shellcheck disable=SC2034
  CMAKE_CUDA_COMPILER=$(which nvcc)

  echo "Installing libjpeg-turbo and libpng"
  yum -y install libjpeg-turbo-devel libpng-devel

  echo "Installing nvJPEG2000"
  url=$(join_by '' \
    "https://developer.download.nvidia.com/compute/nvjpeg2000/0.7.5/" \
    "local_installers/nvjpeg2000-local-repo-rhel9-0.7.5-1.0-1.${ARCH}.rpm")
  wget "$url"
  rpm -i nvjpeg2000-local-repo-rhel9-0.7.5-1.0-1.aarch64.rpm
  dnf clean all
  dnf -y install nvjpeg2k

  echo "Installing torchvision from source"
  url="https://github.com/pytorch/vision/archive/refs/tags/v0.17.0.tar.gz"
  wget "$url"
  tar -xf v0.17.0.tar.gz
  pushd vision-0.17.0 || return 1
  source "$MAIN_VENV_PATH/bin/activate"
  python3 setup.py install
  popd || return 1
}

function install_cleanup() {
  # ec2-user should own the virtual environment
  chown -R "$USER" "$MAIN_VENV_PATH"
}

function install_components() {
  set -v
  # use the larger root volume for temp files during script execution
  mkdir -p "$USER_HOME/tmp"
  pushd "$USER_HOME/tmp" || {
    echo "Failed to enter $USER_HOME/tmp"
    exit 1
  }

  for req in "${@}"; do
    echo "Installing $req"
    mkdir -p "$req"
    pushd "$req" || {
      echo "Failed to enter $req"
      continue
    }
    "install_$req"
    popd || {
      echo "Failed to exit $req"
      exit 1
    }
  done

  popd || echo "Failed to exit $USER_HOME/tmp"
  set +v
}

function install_all_g5g() {
  install_components core_packages yadm gflags_from_source glog_from_source \
    ffmpeg python python_packages nvidia docker pytorch_from_source \
    torchvision_from_source prometheus grafana fluent_bit loki github cleanup
}

function install_all_g5() {
  install_components core_packages yadm awscli gflags_from_source \
    glog_from_source ffmpeg python_packages pytorch prometheus grafana \
    fluent_bit loki github cleanup
}
