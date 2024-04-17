#!/usr/bin/env bash
# https://github.com/ray-project/ray/blob/master/doc/azure/azure-init.sh

if [ $# -gt 1 ]; then
  echo "Usage: $0 [service_user]"
  exit 1
elif [ $# -eq 1 ]; then
  user=$1
else
  user=$USER
fi

log_dir="/tmp/ray/session_latest/artifacts"
script_path="/home/$user/tensorboard.sh"

cat >"$script_path" <<EOM
#!/usr/bin/env bash
mkdir -p $log_dir
tensorboard --port 6006 --logdir=$log_dir
EOM

chmod +x "$script_path"

cat >/etc/systemd/system/tensorboard.service <<EOM
[Unit]
   Description=TensorBoard

[Service]
   Type=simple
   User=$user
   ExecStart=/bin/bash -l $script_path

[Install]
WantedBy=multi-user.target
EOM
