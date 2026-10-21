#!/usr/bin/env bash
set -euo pipefail
trap 'echo "INSTALL_FAILED line=$LINENO (no success claimed)" >&2' ERR
src=$(cd "$(dirname "$0")/.." && pwd)
P45_HOME=${P45_HOME:-/root/autodl-tmp/p45}
export P45_HOME
if [[ ! -d $(dirname "$P45_HOME") ]]; then echo 'Data parent missing; set P45_HOME to a verified data disk' >&2; exit 1; fi
mkdir -p "$P45_HOME" "$P45_HOME/bin" "$P45_HOME/envs"
findmnt -T "$P45_HOME"
available=$(df -PB1 "$P45_HOME" | awk 'NR==2 {print $4}')
if (( available < 3000000000 )); then echo 'Need at least 3GB free on data disk' >&2; exit 1; fi
python3 -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required"'
commit=$(git -C "$src" rev-parse HEAD)
if [[ -n $(git -C "$src" status --porcelain --untracked-files=no) ]]; then echo 'Tracked source modified' >&2; exit 1; fi
envdir="$P45_HOME/envs/$commit"
if [[ ! -x "$envdir/bin/python" ]]; then python3 -m venv --system-site-packages "$envdir"; fi
if ! "$envdir/bin/python" -c 'import torch,numpy; assert torch.__version__.split("+")[0]=="2.14.1" and numpy.__version__=="2.3.5"'; then
  "$envdir/bin/python" -m pip install 'torch==2.14.1' 'numpy==2.3.5' --index-url https://download.pytorch.org/whl/cpu
fi
export PYTHONPATH="$src" PYTHONDONTWRITEBYTECODE=1
"$envdir/bin/python" "$src/scripts/p45.py" check > "$P45_HOME/install-check.json"
# A repeat install may refresh the launcher, but never touches runs or their budgets.
python3 - "$P45_HOME" "$envdir" "$src" <<'PY'
import pathlib,shlex,sys
home,venv,src=sys.argv[1:]
p=pathlib.Path(home)/'bin/p45'
p.write_text('#!/bin/bash\nset -euo pipefail\nexport P45_HOME='+shlex.quote(home)+'\nexport PYTHONPATH='+shlex.quote(src)+'\nexec '+shlex.quote(venv+'/bin/python')+' '+shlex.quote(src+'/scripts/p45.py')+' "$@"\n')
p.chmod(0o755)
PY
echo "INSTALL_OK commit=$commit device=cpu home=$P45_HOME"
echo "Command: $P45_HOME/bin/p45"
