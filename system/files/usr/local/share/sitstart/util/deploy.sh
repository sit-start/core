#!/usr/bin/env bash
shopt -s expand_aliases

# Source relevant util scripts.
for _util in general git py_env; do
  . "$(dirname "${BASH_SOURCE[0]}")/$_util.sh"
done
unset _util

# Use GIT_SSH_COMMAND=$GIT_DEPLOY_SSH_COMMAND to use repo-specific deploy keys.
GIT_DEPLOY_SSH_COMMAND="$(realpath \
  "$SITSTART_SYSTEM_ROOT/usr/local/bin/git_ssh_with_custom_key.sh")"
export GIT_DEPLOY_SSH_COMMAND

# A shorthand prefix for the above, for use with git and yadm.
# shellcheck disable=SC2139
alias with_deploy_key="GIT_SSH_COMMAND=$GIT_DEPLOY_SSH_COMMAND"

# Get the path to the deploy key for a given repo.
function deploy_key_path() {
  local repo_url key_name
  repo_url="$1"
  key_name=$(python3 -c "import os;\
    print(os.path.splitext('${repo_url}'.split(':')[-1].replace('/','-'))[0])")
  echo ~/.ssh/git-keys/"$key_name"
}

# Get the name of the secret for a given repo.
function deploy_key_secrets_name() {
  local repo_url key_path
  repo_url="$1"
  key_path=$(deploy_key_path "$repo_url")
  echo "$(basename "$key_path")-deploy-ed25519"
}

# Create a deploy key for a given repo.
function create_deploy_key() {
  local repo_url key_path key_secrets_name _private_key private_key \
    public_key secret existing_deploy_key_ids
  repo_url="$1"
  key_path=$(deploy_key_path "$repo_url")
  key_secrets_name=$(deploy_key_secrets_name "$repo_url")

  # create the key, without comments
  mkdir -p "$(dirname "$key_path")"
  (ssh-keygen -q -t ed25519 -f "$key_path" -P "" &&
    ssh-keygen -c -C "" -f "$key_path" <<<y >/dev/null 2>&1) ||
    (
      echo "Failed to generate key"
      return 1
    )

  # add the key to AWS Secrets Manager
  _private_key=$(cat "$key_path")
  private_key=${_private_key//$'\n'/\\n}\\n # escape newlines
  public_key=$(xargs <"$key_path".pub)      # trim whitespace
  secret=$(
    join_by '' \
      "{\"private\":\"$private_key\"," \
      "\"public\":\"$public_key\"}"
  )
  aws secretsmanager create-secret --name "$key_secrets_name" \
    --secret-string "$secret" >/dev/null 2>&1 ||
    aws secretsmanager update-secret --secret-id "$key_secrets_name" \
      --secret-string "$secret" >/dev/null 2>&1 ||
    (echo "Failed to update AWS Secrets Manager" && return 1)

  # add the key as a read-only deploy key on GitHub
  # shellcheck disable=SC2207
  existing_deploy_key_ids=($(gh repo deploy-key list -R "$url" --json "id,key" \
    -q ".[] | select(.title==\"$key_secrets_name\") | .id"))
  if [ ${#existing_deploy_key_ids[@]} -gt 0 ]; then
    printf '%s' "Warning - deploy key(s) with name $key_secrets_name " \
      "exist. Delete (y/n)? "
    read -r delete_existing_keys
    if [ "$delete_existing_keys" == "y" ]; then
      for id in "${existing_deploy_key_ids[@]}"; do
        gh repo deploy-key delete -R "$repo_url" "$id"
      done
    fi
  fi
  gh repo deploy-key add -R "$repo_url" -t "$key_secrets_name" "$key_path".pub
}

# Fetch the deploy key from AWS Secrets Manager for a given repo.
function fetch_deploy_key() {
  local repo_url key_path key_secrets_name
  repo_url="$1"
  key_path=$(deploy_key_path "$repo_url")
  key_secrets_name=$(deploy_key_secrets_name "$repo_url")

  mkdir -p "$(dirname "$key_path")"
  test -f "$key_path" ||
    (aws secretsmanager get-secret-value --secret-id "$key_secrets_name" \
      --query SecretString --output text |
      jq --raw-output .private >"$key_path" &&
      chmod go-rwx "$key_path" &&
      echo "Fetched deploy key $key_path for $repo_url.") ||
    (echo "Failed to fetch deploy key for $repo_url." && return 1)
}

# Deploy a yadm (dotfile) repo using deploy keys.
function deploy_yadm_repo() {
  local repo_url repo yadm_config
  if [ "$#" -ne 1 ]; then
    echo "Usage: deploy_yadm_repo <repo_url>"
    return 1
  fi
  repo_url="$1"
  repo=$(basename "$repo_url" .git)
  fetch_deploy_key "$repo_url"
  if yadm_repo_exists >/dev/null 2>&1; then
    echo "Yadm repo already exists, skipping clone."
    return 1
  fi
  with_deploy_key yadm clone "$repo_url" --no-checkout
  yadm_config="$HOME/.local/share/yadm/.git/config"
  yadm config --file "$yadm_config" core.sshCommand "$GIT_DEPLOY_SSH_COMMAND"
  yadm checkout ~
}

# Deploy a repo using deploy keys.
function deploy_repo() {
  local repo_url repo
  if [ "$#" -ne 1 ]; then
    echo "Usage: deploy_repo <repo_url>"
    return 1
  fi
  repo_url="$1"
  repo=$(basename "$repo_url" .git)
  fetch_deploy_key "$repo_url"
  if git_repo_exists "$repo" >/dev/null 2>&1; then
    echo "Repo $repo already exists, skipping clone."
    return 1
  fi
  with_deploy_key git clone "$repo_url"
  git -C "$repo" config --local core.sshCommand "$GIT_DEPLOY_SSH_COMMAND"
}

# Deploy the core repo and update system files and settings.
function deploy_sitstart() {
  # Install the core source repo and use the default venv as local venv.
  mkdir -p "$DEV"
  local core_repo_url=git@github.com:sit-start/core.git
  fetch_deploy_key "$core_repo_url"
  if ! git_repo_exists "$CORE" >/dev/null 2>&1; then
    (cd "$DEV" && deploy_repo $core_repo_url) || return 1
  fi
  if ! [ -d "$CORE/.venv" ]; then
    ln -s "$DEFAULT_VENV" "$CORE/.venv"
  fi

  # Install any missing or out-of-date requirements. TODO: update AMI's
  # default venv with missing requirements (ray upgrade, checksumdir)
  activate_py_env "$CORE/.venv" &&
    pip install --quiet -r "$CORE/requirements-dev.txt" ||
    return 1

  # Deploy system files.
  echo "Deploying system files."
  PYTHONPATH="$CORE/python:$PYTHONPATH" python3 -c "$(
    join_by '; ' \
      'from sitstart.util.system import deploy_system_files' \
      'deploy_system_files("/", as_root=True)'
  )" || return 1

  # Update kernel parameters.
  sudo sysctl -p /etc/sysctl.d/99-sitstart.conf || return 1

  # Install new components not in the AMI. TODO: update the AMI.
  installs=(github)
  if [[ $(uname -s) == "Linux" ]]; then
    sudo bash -c ". $SITSTART_UTIL/install.sh##os.Linux &&\
       install_components ${installs[*]}" || return 1
  else
    echo "Not on Linux, skipping additional component installation." \
      "Ensure you have the following installed: ${installs[*]}."
  fi
}
