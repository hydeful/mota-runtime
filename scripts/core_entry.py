"""P45 CLI with collection-window timing only; no learning logic changes."""
import datetime
import json
from pathlib import Path
import sys
import time
from tasks.tower10.p45.runtime import Runner, atomic_json, cap_deadline

original = Runner.collect

def measured(self, deadline, *args, **kwargs):
    began = time.monotonic()
    cutoff = cap_deadline(deadline, self.s['budget'])
    result = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'requested_window_seconds': max(0, cutoff-began)}
    try:
        return original(self, deadline, *args, **kwargs)
    finally:
        ended = time.monotonic()
        result.update(returned_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), collection_call_seconds=ended-began, post_cutoff_cleanup_seconds=max(0,ended-cutoff))
        atomic_json(self.root/'p48-window.json',result)
        with (self.root/'p48-windows.jsonl').open('a') as f:
            f.write(json.dumps(result)+'\n')
        print(json.dumps({'event':'P48_collection_window',**result}),flush=True)

Runner.collect = measured
from tasks.tower10.p45.cli import main
if len(sys.argv)>1 and sys.argv[1]=='init' and '--mode' not in sys.argv:
    sys.argv.extend(['--mode','serial'])
main()
