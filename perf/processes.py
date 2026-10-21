"""Task-scoped Linux process ownership; no process-name matching or global kill."""
import ctypes
import os
import secrets
import signal
import time
import psutil

class OwnedTask:
    def __init__(self):
        self.token=secrets.token_hex(16);self.known={};self.sid=None;self.denied=set()
        try:self.subreaper=ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0
        except Exception:self.subreaper=False
    def environment(self,base):return dict(base,P48_OWNED_TASK=self.token)
    def attach(self,p):self.sid=p.pid;self.observe()
    def observe(self):
        found=[]
        for p in psutil.process_iter(['pid','create_time','status']):
            try:
                identity=(p.pid,p.create_time())
                # An inherited unguessable task marker also catches descendants
                # that created another session and were reparented before polling.
                owned=identity in self.known
                if not owned:owned=p.environ().get('P48_OWNED_TASK')==self.token
                if owned and p.pid!=os.getpid():
                    self.known[identity]=True;found.append(p)
            except psutil.AccessDenied:self.denied.add(p.pid)
            except (psutil.Error,ProcessLookupError,PermissionError):pass
        return found
    def cleanup(self,grace=.6):
        errors=[];initial=self.observe()
        def row(p):
            try:return {'pid':p.pid,'created':p.create_time(),'status':p.status()}
            except psutil.Error:return {'pid':p.pid,'gone':True}
        before=[row(p) for p in initial]
        for sig in (signal.SIGTERM,signal.SIGKILL):
            procs=self.observe()
            for p in procs:
                try:
                    if p.status()!=psutil.STATUS_ZOMBIE:p.send_signal(sig)
                except psutil.NoSuchProcess:pass
                except psutil.Error as exc:errors.append(str(exc))
            until=time.monotonic()+grace
            while time.monotonic()<until:
                for p in self.observe():
                    try:os.waitpid(p.pid,os.WNOHANG) # only this task's adopted descendants
                    except (ChildProcessError,ProcessLookupError):pass
                if not self.observe():break
                time.sleep(.03)
        remaining=[row(p) for p in self.observe()]
        return {'before_cleanup':before,'remaining':remaining,'complete':not remaining,'errors':errors,'subreaper_enabled':self.subreaper,'unreadable_process_count':len(self.denied),'scope':'observable task-marker processes plus retained identities; inaccessible processes cannot be certified by marker scan','ownership':'unique inherited task marker plus retained PID+creation identities; detached/reparented descendants included; never process names'}
