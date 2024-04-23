#!/usr/bin/env bash
# shellcheck disable=SC2207

_sit_completion() {
  local IFS=$'
'
  COMPREPLY=($(env COMP_WORDS="${COMP_WORDS[*]}" \
    COMP_CWORD="$COMP_CWORD" \
    _SIT_COMPLETE=complete_bash "$1"))
  return 0
}

complete -o default -F _sit_completion sit
