# Sourced by scripts/install_*.sh (ops_robustness_1118). Atomic replacement of deployed files.
#
# Why: bash reads a running script lazily, by byte offset. `cp new qs_job.sh` rewrites the SAME inode in place, so a
# launchd job running qs_job.sh at that moment continues reading the new bytes at the old offset -> "syntax error near
# unexpected token ;;" (launchd-nhl-ingame.err, launchd-nfl-fade.err, 11-15..11-18). Writing a temp file in the same
# directory and `mv -f` onto the path is an atomic rename: the running process keeps the old inode, new runs get the
# new file, nobody ever sees a half-written file. (rsync already does tmp + rename unless --inplace.)
#
#   acp SRC... DEST       like cp: DEST is a directory (each SRC -> DEST/basename) or, with one SRC, a file path
#   awrite DEST [MODE]    stdin -> DEST (heredoc wrappers); MODE default: existing DEST's mode, else 755 for *.sh, 644
_atomic_put() {  # SRC DEST_FILE
  local t
  t="$(mktemp "$(dirname "$2")/.$(basename "$2").tmp.XXXXXX")" || return 1
  if cp "$1" "$t" && chmod "$(stat -f '%Lp' "$1")" "$t" && mv -f "$t" "$2"; then return 0; fi
  rm -f "$t"; return 1
}

acp() {
  [ $# -ge 2 ] || { echo "acp: need SRC... DEST" >&2; return 2; }
  local dest="${!#}" n=$(($# - 1)) i s
  if [ -d "$dest" ]; then
    for ((i = 1; i <= n; i++)); do s="${!i}"; _atomic_put "$s" "${dest%/}/$(basename "$s")" || return 1; done
  else
    [ "$n" -eq 1 ] || { echo "acp: $dest is not a directory" >&2; return 2; }
    _atomic_put "$1" "$dest"
  fi
}

awrite() {
  local dest="$1" mode="${2:-}" t
  if [ -z "$mode" ]; then
    if [ -e "$dest" ]; then mode="$(stat -f '%Lp' "$dest")"; elif [[ "$dest" == *.sh ]]; then mode=755; else mode=644; fi
  fi
  t="$(mktemp "$(dirname "$dest")/.$(basename "$dest").tmp.XXXXXX")" || return 1
  if cat > "$t" && chmod "$mode" "$t" && mv -f "$t" "$dest"; then return 0; fi
  rm -f "$t"; return 1
}
