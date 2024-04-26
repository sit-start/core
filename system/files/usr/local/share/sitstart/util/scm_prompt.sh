#!/usr/bin/env bash
# @source: https://github.com/akushner/dotfiles/blob/master/bin/scm-prompt
# shellcheck disable=SC2162

# Determines the "branch" of the current repo and emits it.
# For use in generating the prompt.
# This is portable to both zsh and bash and works in both
# git and mercurial repos and aims to avoid invoking the
# command line utilities for speed of prompt updates

# To use from zsh:
#  NOTE! the single quotes are important; if you use double quotes
#  then the prompt won't change when you chdir or checkout different
#  branches!
#
#  . /path/to/scm-prompt
#  setopt PROMPT_SUBST
#  export PS1='$(_dotfiles_scm_info)$USER@%m:%~%% '

# To use from bash:
#
#  . /path/to/scm-prompt
#  export PS1="\$(_dotfiles_scm_info)\u@\h:\W\$ "
#
# NOTE! You *EITHER* need to single-quote the whole thing *OR* back-slash
# the $(...) (as above), but not both. Which one you use depends on if
# you need the rest of your PS1 to interpolate variables.

. "$(dirname "${BASH_SOURCE[0]}")/general.sh"

function _dotfiles_scm_info() {
  # find out if we're in a git or hg repo by looking for the control dir
  local d git hg
  d=$PWD
  while :; do
    if test -f "$d/.git"; then
      git="$d"/$(sed -e 's/gitdir\: \(.*\)$/\1/' <"$d/.git")
      git=$(realpath "$git")
      break
    elif test -d "$d/.git"; then
      git=$d/.git
      break
    elif test -d "$d/.hg"; then
      hg=$d/.hg
      break
    fi
    test "$d" = / && break
    d=$(realpath "$d/..")
  done

  local br
  if test -n "$hg"; then
    br=$(test -f "$hg"/dirstate &&
      hexdump -vn 4 -e '1/1 "%02x"' "$hg"/dirstate ||
      echo "empty")
    local file
    # shellcheck disable=SC2066
    for file in "$hg/bookmarks.current"; do
      test -f "$file" && {
        read br <"$file"
        break
      }
    done
    if [ -f "$hg/bisect.state" ]; then
      br="$br|BISECT"
    elif [ -f "$hg/histedit-state" ]; then
      br="$br|HISTEDIT"
    elif [ -f "$hg/graftstate" ]; then
      br="$br|GRAFT"
    elif [ -f "$hg/unshelverebasestate" ]; then
      br="$br|UNSHELVE"
    elif [ -f "$hg/rebasestate" ]; then
      br="$br|REBASE"
    elif [ -d "$hg/merge" ]; then
      br="$br|MERGE"
    fi
  elif test -n "$git"; then
    if test -f "$git/HEAD"; then
      read br <"$git/HEAD"
      case $br in
      ref:\ refs/heads/*) br=${br#ref: refs/heads/} ;;
      # Lop off all of an SHA1 except the leading 7 hex digits.
      # Use this cumbersome notation (it's portabile) rather
      # than ${br:0:7}, which doesn't work with older zsh.
      *) br="${br%????????????????????????????????}" ;;
      esac
      if [ -f "$git/rebase-merge/interactive" ]; then
        b="$(cat "$git/rebase-merge/head-name")"
        b=${b#refs/heads/}
        br="$br|REBASE-i|$b"
      elif [ -d "$git/rebase-merge" ]; then
        b="$(cat "$git/rebase-merge/head-name")"
        b=${b#refs/heads/}
        br="br|REBASE-m|$b"
      else
        if [ -d "$git/rebase-apply" ]; then
          if [ -f "$git/rebase-apply/rebasing" ]; then
            br="$br|REBASE"
          elif [ -f "$git/rebase-apply/applying" ]; then
            br="$br|AM"
          else
            br="$br|AM/REBASE"
          fi
        elif [ -f "$git/MERGE_HEAD" ]; then
          br="$br|MERGE"
        elif [ -f "$git/BISECT_LOG" ]; then
          br="$br|BISECT"
        fi
      fi
    fi
  fi
  # Be compatable with __git_ps1. In particular:
  # - provide a space for the user so that they don't have to have
  #   random extra spaces in their prompt when not in a repo
  # - provide parens so it's differentiated from other crap in their prompt
  if [ -n "$br" ]; then
    if [ -n "$WANT_OLD_SCM_PROMPT" ]; then
      printf %s "$br"
    else
      printf '(%s)' "$br"
    fi
  fi
}
