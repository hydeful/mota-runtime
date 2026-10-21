"""Small comparator regression checks, no collection/training."""
import copy
import random
import unittest
import numpy as np
import torch
from perf.semantic import compare,report
from tasks.tower10.p45.search import Node

class SemanticTests(unittest.TestCase):
    def test_rng_no_advance(self):
        a={'python':random.Random(1),'numpy':np.random.RandomState(2),'actor':np.random.default_rng(3),'torch':torch.Generator().manual_seed(4)}
        before=copy.deepcopy(a);compare(a,copy.deepcopy(a));compare(before,a)
        b=copy.deepcopy(a);b['actor'].random();self.assertIn('actor',report(a,b)['first_difference'])
    def test_tree_and_arrays(self):
        a=Node({'x':np.arange(6)},0);a.children[1]=Node({'leaf':True},1);a.n=np.array([2,3]);b=copy.deepcopy(a);compare(a,b)
        b.children[1].t+=1;self.assertIn('children',report(a,b)['first_difference'])
        b=copy.deepcopy(a);b.n.flags.writeable=False;self.assertIn('writeable',report(a,b)['first_difference'])
        for change in (lambda b:setattr(b,'n',b.n.astype(np.float32)),lambda b:setattr(b,'n',b.n.reshape(1,2))):
            b=copy.deepcopy(a);change(b);self.assertEqual(report(a,b)['status'],'fail')
    def test_model_optimizer_progress_rng(self):
        a={'model':{'w':torch.arange(3.)},'optimizer':{'momentum':torch.ones(3)},'progress':{'steps_done':7},'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}}
        for key in ('model','optimizer','progress','rng'):
            b=copy.deepcopy(a)
            if key=='model':b[key]['w'][0]+=1
            elif key=='optimizer':b[key]['momentum'][0]+=1
            elif key=='progress':b[key]['steps_done']+=1
            else:b[key]['torch'][0]^=1
            self.assertEqual(report(a,b)['status'],'fail')
    def test_unknown_fails(self):
        class Unsupported:pass
        r=report({'unknown':Unsupported()},{'unknown':Unsupported()});self.assertEqual(r['status'],'fail');self.assertIn('UNKNOWN_OBJECT_TYPE',r['first_difference']);self.assertNotIn('0x',r['first_difference'])
if __name__=='__main__':unittest.main()
