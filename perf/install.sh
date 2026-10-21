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
if [[ ! -e "$src" ]]; then
 git init "$src"
 git -C "$src" remote add origin https://github.com/hydeful/mota-runtime.git
fi
[[ -d "$src/.git" ]] || { echo 'Existing non-repository directory refused' >&2; exit 1; }
remote=$(git -C "$src" remote get-url origin 2>/dev/null || true)
[[ $remote == https://github.com/hydeful/mota-runtime.git ]] || { echo 'Unexpected/missing origin refused; no files overwritten' >&2; exit 1; }
if git -C "$src" rev-parse --verify HEAD >/dev/null 2>&1; then
 [[ $(git -C "$src" rev-parse HEAD) == "$commit" ]] || { echo 'Existing different checkout refused' >&2; exit 1; }
 [[ -z $(git -C "$src" status --porcelain --untracked-files=no) ]] || { echo 'Modified tracked source refused' >&2; exit 1; }
else
 # Retry a previously failed fetch only if it has no working/index payload.
 [[ -z $(git -C "$src" ls-files) && -z $(git -C "$src" ls-files --others --exclude-standard) ]] || { echo 'Incomplete checkout has user files; refused' >&2; exit 1; }
 echo 'Downloading/retrying fixed source commit (an existing .git is not completion).'
 GIT_TERMINAL_PROMPT=0 git -C "$src" -c credential.helper= fetch --depth=1 origin "$commit"
 git -C "$src" cat-file -e "$commit^{commit}"
 git -C "$src" fsck --connectivity-only --no-dangling "$commit"
 git -C "$src" checkout --detach "$commit"
fi
[[ $(git -C "$src" rev-parse HEAD) == "$commit" ]] || exit 1
git -C "$src" fsck --connectivity-only --no-dangling "$commit"
[[ -z $(git -C "$src" status --porcelain --untracked-files=no) ]] || { echo 'Modified tracked source refused' >&2; exit 1; }
# Reuse only functionally verified prior interpreters; code revision alone is
# never a reason to download another Torch/CUDA distribution.
prior_cpu=$(python3 -c 'import json,pathlib,sys;p=pathlib.Path(sys.argv[1]);print(json.loads(p.read_text()).get("cpu_python", "") if p.exists() else "")' "$P48_PERF_HOME/installation.json")
cpu="$P48_PERF_HOME/envs/$commit-cpu"
if [[ -x $prior_cpu ]] && "$prior_cpu" -c 'import torch,numpy,psutil;assert str(torch.__version__)=="2.14.1+cpu" and numpy.__version__=="2.3.5";x=torch.ones(2,requires_grad=True);x.square().sum().backward();assert torch.equal(x.grad,torch.full((2,),2.))' >/dev/null 2>&1; then
 cpu=$(dirname "$(dirname "$prior_cpu")")
 echo "Reusing verified CPU environment: $cpu"
fi
[[ -x "$cpu/bin/python" ]] || python3 -m venv --system-site-packages "$cpu"
if ! "$cpu/bin/python" -c 'import torch,numpy;assert str(torch.__version__)=="2.14.1+cpu" and numpy.__version__=="2.3.5"' >/dev/null 2>&1; then
 "$cpu/bin/pip" install 'torch==2.14.1+cpu' 'numpy==2.3.5' --index-url https://download.pytorch.org/whl/cpu
fi
"$cpu/bin/python" -c 'import psutil' >/dev/null 2>&1 || "$cpu/bin/pip" install 'psutil==7.2.2'
gpu_python=''
prior_gpu=$(python3 -c 'import json,pathlib,sys;p=pathlib.Path(sys.argv[1]);print(json.loads(p.read_text()).get("gpu_python") or "" if p.exists() else "")' "$P48_PERF_HOME/installation.json")
if [[ $gpu_mode != off && -x $prior_gpu ]] && "$prior_gpu" -c 'import torch,numpy,psutil;assert str(torch.__version__).split("+")[0] in ("2.14.1","2.11.0") and numpy.__version__=="2.3.5" and torch.cuda.is_available();x=torch.ones(2,device="cuda",requires_grad=True);x.square().sum().backward();torch.cuda.synchronize();assert bool((x.grad==2).all())' >/dev/null 2>&1; then
 gpu_python=$prior_gpu
 echo "Reusing verified CUDA environment: $gpu_python"
fi
if [[ $gpu_mode != off && -z $gpu_python ]] && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
 if (( free < 10000000000 )); then echo 'GPU environment skipped: need>=10GB free' >&2
 else
  gpu="$P48_PERF_HOME/envs/$commit-gpu"
  [[ -x "$gpu/bin/python" ]] || python3 -m venv "$gpu"
  echo "Preparing isolated CUDA environment. Live progress below; persistent log: $P48_PERF_HOME/gpu-install.log"
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  flavor=cu128; version=2.11.0
  # Official2.14.1 CUDA wheels are cu130, not cu128. Driver580+ may use
  # them; otherwise cu128/2.11.0 is isolated and cross-version numerically gated.
  if (( ${driver%%.*} >= 580 )); then flavor=cu130; version=2.14.1; fi
  : > "$P48_PERF_HOME/gpu-install.log"
  gpu_ready() {
   "$gpu/bin/pip" install "torch==$1" 'numpy==2.3.5' --index-url "https://download.pytorch.org/whl/$2" 2>&1 | tee -a "$P48_PERF_HOME/gpu-install.log" &&
   "$gpu/bin/pip" install 'psutil==7.2.2' 2>&1 | tee -a "$P48_PERF_HOME/gpu-install.log" &&
   "$gpu/bin/python" -c 'import torch,numpy as np; assert torch.cuda.is_available() and torch.version.cuda, "CUDA environment unusable"; assert torch.from_numpy(np.ones(1,np.float32)).numpy()[0]==1' 2>&1 | tee -a "$P48_PERF_HOME/gpu-install.log"
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
s=root/'bin/mota-smoke';s.write_text(p.read_text().replace('perf.control','perf.smoke'));s.chmod(0o755)
PY
"$cpu/bin/python" -B -m perf.semantic_test
"$cpu/bin/python" -B -m perf.control check
"$cpu/bin/python" -B -m perf.selftest --out "$P48_PERF_HOME/checks/$commit-$(date +%s)"
"$cpu/bin/python" - "$P48_PERF_HOME/installation.json" "$started" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);x=json.loads(p.read_text());x['install_seconds']=time.time()-int(sys.argv[2]);x['small_selftest_pass']=True;p.write_text(json.dumps(x,indent=2)+'\n')
PY
echo "PERF_INSTALL_OK commit=$commit; CPU baseline preserved; GPU=${gpu_python:-unavailable}; no experiment started"
echo "Command: $P48_PERF_HOME/bin/mota-perf"
