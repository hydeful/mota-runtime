"""P41 T1 candidate: P36 interaction boundary, original 713 model action IDs."""
import numpy as np
from tasks.tower10.neural.common import RuleMask
from .actions import interaction_kind,require_progress

class InteractionMask(RuleMask):
    def __call__(self,state):
        mask=super().__call__(state)
        for i in np.flatnonzero(mask):
            if interaction_kind(state,self.codec.actions[int(i)]) is None:mask[i]=False
        return mask

class InteractionSession:
    def __init__(self,session):self.raw=session;self.initial=session.initial
    def step(self,state,action):
        if interaction_kind(state,action) is None:raise ValueError('NON_INTERACTION_ACTION '+repr(action))
        result=self.raw.step(state,action)
        require_progress(state,action,result)
        return result
    def compile_actions(self,actions,state=None):
        state=self.initial if state is None else state
        route=[];records=[]
        for a in actions:
            r=self.step(state,a);state=r['state'];route.extend(r['routeSegment']);records.append(r['result'])
        return dict(state=state,route=route,records=records)
