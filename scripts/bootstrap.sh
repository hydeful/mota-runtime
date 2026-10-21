#!/usr/bin/env bash
# Usage: bash bootstrap.sh <full-public-commit> [data-home]
set -euo pipefail
trap 'echo "DOWNLOAD_OR_INSTALL_FAILED line=$LINENO" >&2' ERR
commit=${1:?Pass the fixed full public commit from the delivery}
[[ $commit =~ ^[0-9a-f]{40}$ ]] || { echo 'Full 40-character commit required' >&2; exit 1; }
export P45_HOME=${2:-/root/autodl-tmp/p45}
[[ -d $(dirname "$P45_HOME") ]] || { echo 'Missing data disk parent' >&2; exit 1; }
mkdir -p "$P45_HOME/code"
src="$P45_HOME/code/$commit"
if [[ ! -d "$src/.git" ]]; then
  [[ ! -e "$src" ]] || { echo 'Existing non-git directory; refusing overwrite' >&2; exit 1; }
  git init "$src"
  git -C "$src" remote add origin https://github.com/hydeful/mota-p45-runtime.git
  GIT_TERMINAL_PROMPT=0 git -C "$src" -c credential.helper= fetch --depth=1 origin "$commit"
  git -C "$src" checkout --detach FETCH_HEAD
fi
[[ $(git -C "$src" rev-parse HEAD) == "$commit" ]] || { echo 'Commit mismatch' >&2; exit 1; }
bash "$src/scripts/install.sh"
