#!/usr/bin/env bash

# Join strings with a separator. Usage: join_by sep tok1 tok2 ... tokn.
function join_by {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

# Notify when a command is done.
function notify_done() {
  if [[ $(uname -s) != "Darwin" ]]; then
    echo "This function is only available on macOS."
    return 1
  fi
  cmd="$*"
  $cmd
  osascript -e \
    "display notification \
      \"Command: $cmd\nExit status: $?\" with title \"Done\""
}

# A portable "realpath" equivalent.
if ! [ "$(command -v realpath)" ]; then
  function realpath() {
    cd "$1" && echo "$PWD"
  }
fi
