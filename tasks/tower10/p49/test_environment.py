import copy
import json
import unittest
from .environment import Environment, build, reference, HERE
from .certify import replay


class MinimalTowerTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment()

    def test_map_and_original_rule_data(self):
        s = self.env.initial
        self.assertEqual(len(s['graph']['nodes']), 31)
        self.assertEqual(len(s['objects']), 21)
        self.assertEqual(sum(o['event']['id']=='redSlime' for o in s['objects'].values()),3)
        self.assertEqual(s['hero']['items']['tools'], {})
        self.assertEqual(len(self.env.legal(s)),4)
        original=json.loads((HERE.parent/'p45/assets/shared_initial_topology.json').read_text())
        for key, enemy in s['rules']['enemies'].items():
            self.assertEqual(enemy, original['rules']['enemies'][key])
        for key, item in s['rules']['items'].items():
            self.assertEqual(item, original['rules']['items'][key])
        for row in s['graph']['out'].values():
            for e in row:
                a,b=e['path'];self.assertEqual(abs(a[1]-b[1])+abs(a[2]-b[2]),1)

    def test_mask_and_obstruction_are_original(self):
        s=self.env.initial
        self.assertEqual(reference.refusal(s,'o:MT1:4:1'),'CANNOT_BATTLE')
        self.assertEqual(reference.refusal(s,'o:MT1:2:4'),'INSUFFICIENT_KEY')
        self.assertEqual(reference.refusal(s,'o:MT1:0:4'),'NOT_NEXT_OBJECT')
        self.assertEqual(reference.damage(s,'bluePriest'),638)
        action={'target':'o:MT1:4:1'}
        out=self.env.step(s,action)
        self.assertFalse(out['result']['accepted']);self.assertEqual(s,out['state'])

    def test_door_consumption_no_pass_and_no_repetition(self):
        s=self.env.initial
        s=self.env.step(s,{'target':'o:MT1:3:3'})['state']
        before=copy.deepcopy(s)
        s=self.env.step(s,{'target':'o:MT1:2:4'})['state']
        self.assertEqual(before['hero']['items']['tools'],{'yellowKey':1})
        self.assertEqual(s['hero']['items']['tools'],{})
        self.assertEqual((s['hero']['x'],s['hero']['y']),(3,4))
        self.assertEqual(reference.refusal(s,'o:MT1:2:4'),'OBJECT_NOT_ACTIVE')
        self.assertEqual(s['entities'][s['objects']['o:MT1:2:4']['entity']]['status'],'resolved')

    def test_strict_survival_boundary(self):
        route=['o:MT1:3:3','o:MT1:2:4','o:MT1:1:4','o:MT1:0:4']
        s=self.env.initial
        for k in route:s=self.env.step(s,{'target':k})['state']
        self.assertEqual(s['hero']['atk'],20)
        self.assertEqual(reference.damage(s,'skeletonSoldier'),252)
        s=self.env.raw.export(s);s['hero']['hp']=252
        self.assertEqual(reference.refusal(s,self.env.boss),'WOULD_DIE')
        s['hero']['hp']=253
        out=self.env.step(s,{'target':self.env.boss})
        self.assertEqual(self.env.terminal(out['state']),{'end':'GOAL','score':1})

    def test_full_reference_replay_and_atomic_goal(self):
        route=[{'target':k} for k in ['o:MT1:3:3','o:MT1:2:4','o:MT1:1:4','o:MT1:0:4','o:MT1:4:1']]
        self.assertEqual(replay(self.env,route)['terminal'],{'end':'GOAL','score':98})

    def test_browser_bridge_same_snapshots_and_transitions(self):
        # Execute the exact browser bridge on CPython against the same engine.
        import sys
        import types
        original = sys.modules.get('topology_engine')
        sys.modules['topology_engine']=reference
        bridge=(HERE/'web/src/bridge.py').read_text().replace("json.load(open('/initial.json'))",'provided_initial')
        ns={'provided_initial':build()}
        try:
            exec(compile(bridge,'bridge.py','exec'),ns)
            s=self.env.initial
            route=['o:MT1:3:3','o:MT1:2:4','o:MT1:1:4','o:MT1:0:4','o:MT1:4:1']
            for key in route:
                s=self.env.step(s,{'target':key})['state']
                v=json.loads(ns['interact'](key))
                self.assertEqual(s,ns['state'])
                self.assertEqual(v['actions'],self.env.legal(s))
                self.assertEqual(v['terminal'],self.env.terminal(s))
        finally:
            if original is None:sys.modules.pop('topology_engine',None)
            else:sys.modules['topology_engine']=original


if __name__ == '__main__':
    unittest.main()
