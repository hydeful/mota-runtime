#!/usr/bin/env bash
# Standalone public installer: bash install.sh FULL_COMMIT [auto|required|off]
set -euo pipefail
trap 'echo "PERF_INSTALL_FAILED line=$LINENO; old CPU entry untouched" >&2' ERR
commit=${1:?Full public commit required}
gpu_mode=${2:-auto}
[[ $commit =~ ^[0-9a-f]{40}$ ]] || { echo 'Full40-character commit required' >&2; exit 1; }
[[ $gpu_mode == auto || $gpu_mode == required || $gpu_mode == off ]] || exit 2
P48_PERF_HOME=${P48_PERF_HOME:-/root/autodl-tmp/p48-perf}
export P48_PERF_HOME
[[ -d $(dirname "$P48_PERF_HOME") ]] || { echo 'Data disk parent missing' >&2; exit 1; }
mkdir -p "$P48_PERF_HOME/code" "$P48_PERF_HOME/envs" "$P48_PERF_HOME/bin"
findmnt -T "$P48_PERF_HOME"
if [[ $P48_PERF_HOME == /root/autodl-tmp/p48-perf && $(findmnt -n -o TARGET -T "$P48_PERF_HOME") == / ]]; then echo 'AutoDL data disk not confirmed' >&2; exit 1; fi
free=$(df -PB1 "$P48_PERF_HOME" | awk 'NR==2{print $4}')
(( free >= 3000000000 )) || { echo 'Need>=3GB free for CPU install' >&2; exit 1; }
python3 -c 'import sys; assert sys.version_info[:2]==(3,12), "Python3.12 required"'
started=$(date +%s)
src="$P48_PERF_HOME/code/$commit"
if [[ ! -d "$src/.git" ]]; then
 [[ ! -e "$src" ]] || { echo 'Existing directory refused' >&2; exit 1; }
 git init "$src"
 git -C "$src" remote add origin https://github.com/hydeful/mota-runtime.git
 GIT_TERMINAL_PROMPT=0 git -C "$src" -c credential.helper= fetch --depth=1 origin "$commit"
 git -C "$src" checkout --detach FETCH_HEAD
fi
[[ $(git -C "$src" rev-parse HEAD) == "$commit" ]] || exit 1
[[ -z $(git -C "$src" status --porcelain --untracked-files=no) ]] || { echo 'Modified tracked source refused' >&2; exit 1; }
cpu="$P48_PERF_HOME/envs/$commit-cpu"
[[ -x "$cpu/bin/python" ]] || python3 -m venv --system-site-packages "$cpu"
if ! "$cpu/bin/python" -c 'import torch,numpy;assert str(torch.__version__)=="2.14.1+cpu" and numpy.__version__=="2.3.5"' >/dev/null 2>&1; then
 "$cpu/bin/pip" install 'torch==2.14.1+cpu' 'numpy==2.3.5' --index-url https://download.pytorch.org/whl/cpu
fi
"$cpu/bin/python" -c 'import psutil' >/dev/null 2>&1 || "$cpu/bin/pip" install 'psutil==7.2.2'
gpu_python=''
if [[ $gpu_mode != off ]] && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
 if (( free < 10000000000 )); then echo 'GPU environment skipped: need>=10GB free' >&2
 else
  gpu="$P48_PERF_HOME/envs/$commit-gpu"
  [[ -x "$gpu/bin/python" ]] || python3 -m venv "$gpu"
  echo 'Preparing isolated CUDA environment; CPU baseline and cloud600 are untouched.'
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  flavor=cu128; version=2.11.0
  # Official2.14.1 CUDA wheels are cu130, not cu128. Driver580+ may use
  # them; otherwise cu128/2.11.0 is isolated and cross-version numerically gated.
  if (( ${driver%%.*} >= 580 )); then flavor=cu130; version=2.14.1; fi
  : > "$P48_PERF_HOME/gpu-install.log"
  gpu_ready() {
   "$gpu/bin/pip" install "torch==$1" 'numpy==2.3.5' --index-url "https://download.pytorch.org/whl/$2" >> "$P48_PERF_HOME/gpu-install.log" 2>&1 &&
   "$gpu/bin/pip" install 'psutil==7.2.2' >> "$P48_PERF_HOME/gpu-install.log" 2>&1 &&
   "$gpu/bin/python" -c 'import torch,numpy as np; assert torch.cuda.is_available() and torch.version.cuda, "CUDA environment unusable"; assert torch.from_numpy(np.ones(1,np.float32)).numpy()[0]==1' >> "$P48_PERF_HOME/gpu-install.log" 2>&1
  }
  if gpu_ready "$version" "$flavor"; then gpu_python="$gpu/bin/python"
  elif [[ $flavor == cu130 ]] && gpu_ready 2.11.0 cu128; then gpu_python="$gpu/bin/python"
  else echo 'GPU environment unavailable; GPU cases will be reported skipped; see gpu-install.log' >&2
  fi
 fi
fi
if [[ $gpu_mode == required && -z $gpu_python ]]; then echo 'Required GPU environment not ready; experiment NOT started' >&2; exit 1; fi
export PYTHONPATH="$src" PYTHONDONTWRITEBYTECODE=1
"$cpu/bin/python" - "$P48_PERF_HOME" "$cpu" "$gpu_python" "$src" "$started" <<'PY'
import json,pathlib,shlex,sys,time,subprocess
home,cpu,gpu,src,started=sys.argv[1:];root=pathlib.Path(home)
gpu_info=json.loads(subprocess.check_output([gpu,'-c','import json,torch;print(json.dumps({"torch":str(torch.__version__),"cuda":torch.version.cuda,"available":torch.cuda.is_available()}))'],text=True)) if gpu else None
(root/'installation.json').write_text(json.dumps({'source':src,'cpu_python':cpu+'/bin/python','gpu_python':gpu or None,'gpu_software':gpu_info,'install_seconds':time.time()-int(started),'gpu_install_log':str(root/'gpu-install.log'),'installation_outside_experiment_budget':True},indent=2)+'\n')
p=root/'bin/mota-perf'
p.write_text('#!/bin/bash\nset -euo pipefail\nexport P48_PERF_HOME='+shlex.quote(home)+'\nexport PYTHONPATH='+shlex.quote(src)+'\nexec '+shlex.quote(cpu+'/bin/python')+' -B -m perf.control "$@"\n');p.chmod(0o755)
PY
"$cpu/bin/python" -B -m perf.control check
"$cpu/bin/python" -B -m perf.selftest --out "$P48_PERF_HOME/checks/$commit-$(date +%s)"
"$cpu/bin/python" - "$P48_PERF_HOME/installation.json" "$started" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);x=json.loads(p.read_text());x['install_seconds']=time.time()-int(sys.argv[2]);x['small_selftest_pass']=True;p.write_text(json.dumps(x,indent=2)+'\n')
PY
echo "PERF_INSTALL_OK commit=$commit; CPU baseline preserved; GPU=${gpu_python:-unavailable}; no experiment started"
echo "Command: $P48_PERF_HOME/bin/mota-perf"
