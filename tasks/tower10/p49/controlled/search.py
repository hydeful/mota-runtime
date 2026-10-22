"""Original selection and backup; read-only logging after original finish."""
import numpy as np
from tasks.tower10.p49.learning.search import Search as OriginalSearch,Node


def resources(s):
    h=s['hero'];return {'hp':h['hp'],'atk':h['atk'],'def':h['def'],
                       'yellowKey':h['items'].get('tools',{}).get('yellowKey',0),'money':h['money']}


class Search(OriginalSearch):
    def finish(self,root,decision,rng):
        child,record=super().finish(root,decision,rng)
        record.update(network_prior=root.prior.tolist(),
          added_N=(root.n-decision['inherited_N']).tolist(),
          action_Q=np.divide(root.w,root.n,out=np.zeros_like(root.w),where=root.n>0).tolist(),
          resources_before=resources(root.state),resources_after=resources(child.state))
        record['resource_delta']={k:record['resources_after'][k]-v for k,v in record['resources_before'].items()}
        return child,record
