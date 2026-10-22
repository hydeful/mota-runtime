"""P49 34x5x9 encoding; original x/y coordinates, no fabricated map objects."""
import numpy as np


class Observation:
    statuses = ['absent', 'active', 'hidden', 'resolved', 'moved']
    def __init__(self, initial, h_bound):
        self.types = sorted({o['event']['id'] for o in initial['objects'].values()})
        self.channels = ['walkable', 'wall', 'barrier', 'hero'] + ['type:'+v for v in self.types] + ['status:'+v for v in self.statuses] + ['enemy_hp', 'enemy_atk', 'enemy_def', 'door_key_cost', 'item_hp', 'item_atk', 'item_def', 'item_key', 'hp', 'atk', 'def', 'yellowKey', 'goal', 't']
        assert len(self.channels) == 34
        self.index = {v:i for i,v in enumerate(self.channels)}
        self.h_bound = h_bound
        self.walk = np.zeros((5,9), np.float32)
        for n in initial['graph']['nodes'].values():
            _, x, y = n['position']; self.walk[y,x] = 1

    def encode(self, s, t):
        x = np.zeros((len(self.channels),5,9),np.float32); idx=self.index
        x[idx['walkable']] = self.walk; x[idx['wall']] = 1-self.walk
        h=s['hero'];x[idx['hero'],h['y'],h['x']]=1
        x[idx['status:absent']]=1
        for o in s['objects'].values():
            _,cx,cy=o['position'];typ=o['event']['id']
            x[idx['type:'+typ],cy,cx]=1
            x[idx['status:absent'],cy,cx]=0;x[idx['status:'+o['status']],cy,cx]=1
            if o['trigger']=='battle':
                enemy=s['rules']['enemies'][typ]
                for stat in ['hp','atk','def']:x[idx['enemy_'+stat],cy,cx]=enemy[stat]/100
            elif o['trigger']=='openDoor':
                x[idx['door_key_cost'],cy,cx]=o['event']['doorInfo']['keys'].get('yellowKey',0)
            elif o['trigger']=='getItem':
                if typ in ('redPotion','bluePotion'):x[idx['item_hp'],cy,cx]=s['rules']['values'][typ]*s['rules']['ratios'][o['position'][0]]/self.h_bound
                elif typ=='sword1':x[idx['item_atk'],cy,cx]=.5
                elif typ=='shield1':x[idx['item_def'],cy,cx]=.5
                elif typ=='yellowKey':x[idx['item_key'],cy,cx]=1
        for k in s['barriers']:
            _,_,cx,cy=k.split(':');x[idx['barrier'],int(cy),int(cx)]=1
        for stat,scale in [('hp',self.h_bound),('atk',20),('def',20)]:x[idx[stat]]=h[stat]/scale
        x[idx['yellowKey']]=h['items'].get('tools',{}).get('yellowKey',0)/4
        x[idx['goal']]=s['task']['status']=='GOAL';x[idx['t']]=t/21
        if not np.isfinite(x).all():raise ValueError('NONFINITE_OBSERVATION')
        return x
