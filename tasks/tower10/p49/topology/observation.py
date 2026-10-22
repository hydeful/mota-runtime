"""Object features, object relations and hero scalars; no grid or coordinates."""
import numpy as np
from .kernel import component_and_frontier

NODE_FEATURES=27
RELATIONS=2
GLOBAL_FEATURES=7


class Observation:
    def __init__(self,initial):
        self.order=list(initial['graph']['order']);self.types=sorted({o['event']['id'] for o in initial['objects'].values()})
        self.statuses=['absent','active','hidden','resolved','moved']
        self.channels=['type:'+k for k in self.types]+['status:'+k for k in self.statuses]+['enemy_hp','enemy_atk','enemy_def','yellow_key_cost','pickup_hp','pickup_atk','pickup_def','pickup_key','frontier','source_component','goal_object']
        assert len(self.channels)==NODE_FEATURES
        self.index={k:i for i,k in enumerate(self.channels)};self.n=len(self.order)
        self.adjacency=np.array([[float(j in initial['graph']['adjacency'][i]) for j in self.order] for i in self.order],np.float32)

    def tensors(self,s,t):
        idx=self.index;nodes=np.zeros((self.n,NODE_FEATURES),np.float32)
        component,frontier=component_and_frontier(s);resolved=[]
        for i,key in enumerate(self.order):
            o=s['objects'][key];typ=o['event']['id'];nodes[i,idx['type:'+typ]]=1;nodes[i,idx['status:'+o['status']]]=1
            resolved.append(o['status']!='active')
            if o['trigger']=='battle':
                for stat in ['hp','atk','def']:nodes[i,idx['enemy_'+stat]]=s['rules']['enemies'][typ][stat]/100
            if o['trigger']=='openDoor':nodes[i,idx['yellow_key_cost']]=o['event']['doorInfo']['keys'].get('yellowKey',0)
            if o['trigger']=='getItem':
                if typ in ['redPotion','bluePotion']:nodes[i,idx['pickup_hp']]=s['rules']['values'][typ]/900
                elif typ=='sword1':nodes[i,idx['pickup_atk']]=.5
                elif typ=='shield1':nodes[i,idx['pickup_def']]=.5
                elif typ=='yellowKey':nodes[i,idx['pickup_key']]=1
            nodes[i,idx['frontier']]=key in frontier;nodes[i,idx['source_component']]=key in component;nodes[i,idx['goal_object']]=key==s['task']['boss']
        # R0: potential direct corridor adjacency. R1: adjacency whose target
        # no longer blocks passage. Resource guards live in object attributes
        # and global resources, and are enforced by the original rule refusal.
        relations=np.stack([self.adjacency,self.adjacency*np.array(resolved,np.float32)[None,:]])
        h=s['hero'];hero=np.array([h['hp']/900,h['atk']/20,h['def']/20,h['items']['tools'].get('yellowKey',0)/4,h['money']/100,t/21,float(s['task']['status']=='GOAL')],np.float32)
        return nodes,relations,hero

    def encode(self,s,t):
        nodes,relations,hero=self.tensors(s,t)
        # Serialization envelope for the unchanged search/SGD tensor interfaces.
        # Rows are objects; relation columns are object endpoints, not cells.
        return np.concatenate([nodes,relations.transpose(1,0,2).reshape(self.n,-1),np.broadcast_to(hero,(self.n,GLOBAL_FEATURES))],axis=1).astype(np.float32)

    @staticmethod
    def permute(packed,permutation):
        p=np.asarray(permutation);n=len(p);nodes=packed[:,:NODE_FEATURES][p]
        relations=packed[:,NODE_FEATURES:NODE_FEATURES+RELATIONS*n].reshape(n,RELATIONS,n).transpose(1,0,2)
        relations=relations[:,p][:,:,p]
        return np.concatenate([nodes,relations.transpose(1,0,2).reshape(n,-1),packed[:,-GLOBAL_FEATURES:][p]],axis=1)
