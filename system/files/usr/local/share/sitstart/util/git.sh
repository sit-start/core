#!/usr/bin/env bash

util_path=$(dirname "${BASH_SOURCE[0]}")
. "$util_path/general.sh"

function git_is_clean() {
  if [ -n "$(git status --porcelain)" ]; then
    msg=$(join_by ' ' \
      "You have uncommitted changes." \
      "You may want to commit or stash them first.")
    echo "$msg"
    return 1
  fi
  return 0
}

function git_is_synced() {
  # @source: https://gist.github.com/leroix/8829846
  local local_branch=${1:-$(git rev-parse --abbrev-ref HEAD)}
  local remote_branch="${2:-origin}/$local_branch"
  local local_rev
  local_rev=$(git rev-parse "$local_branch")
  local remote_rev
  remote_rev=$(git rev-parse "$remote_branch")

  if [ x"$local_rev" != x"$remote_rev" ]; then
    msg=$(join_by ' ' \
      "Your local branch $local_branch is not in sync with" \
      "the remote branch $remote_branch." \
      "You may want to rebase or push first.")
    echo "$msg"
    return 1
  fi
  return 0
}

function git_repo_exists() {
  if (git -C "$1" rev-parse --is-inside-work-tree) >/dev/null 2>&1; then
    echo "Repository exists at $1"
    return 0
  else
    echo "No repository exists at $1"
    return 1
  fi
}

function yadm_repo_exists() {
  if [ -d "$HOME/.local/share/yadm/repo.git" ]; then
    echo "Yadm repository exists for user '$USER'."
    return 0
  else
    echo "No yadm repository exists for user '$USER'."
    return 1
  fi
}
