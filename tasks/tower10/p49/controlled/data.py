"""Offline preparation: replay actual actions in both frozen engines, no answers."""
import gzip,hashlib,json,shutil
from pathlib import Path
import numpy as np
from tasks.tower10.neural.common import digest
from tasks.tower10.p49.learning.environment import Environment as OldMap
from tasks.tower10.p49.topology.environment import Environment as OldTopology
from .environment import Environment,make_config,assert_configs
from .observation import unpack,SLICES


def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def prepare(destination,map_run,topology_run):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    a=Environment(make_config('A',phase='fit'));b=Environment(make_config('B',phase='fit'))
    assert_configs(a.config,b.config)
    olda=OldMap();oldb=OldTopology()
    mapping=json.loads((Path(__file__).resolve().parent.parent/'topology/display_mapping.json').read_text())
    bynode={m['node']:m['reference_slot'] for m in mapping}
    dictionary=[{'id':i,'A':act['target'],'B':b.codec.actions[i]['target']} for i,act in enumerate(a.codec.actions)]
    assert all(bynode[x['B']]==x['A'] for x in dictionary)
    write(destination/'actions.json',dictionary);action_sha=digest(dictionary)
    def projected(s,physical):
        return {'hero':{k:s['hero'][k] for k in ['hp','atk','def','money','items','experience']},
          'objects':{(next(x['B'] for x in dictionary if x['A']==k) if physical else k):o['status'] for k,o in s['objects'].items()},
          'goal':s['task']['status'],'weapon':s['flags'].get('nowWeapon'),
          'shield':s['flags'].get('nowShield'),'hatred':s['flags'].get('hatred',0)}
    xs={'A':[],'B':[]};masks=[];pis=[];zs=[];held=[];samples=[];sources=[];roundtrips=0
    for origin,root,splitpoint,old in [('map',Path(map_run),24,olda),('topology',Path(topology_run),20,oldb)]:
        status=json.loads((root/'chain/status.json').read_text())
        for brief in status['episodes']:
            path=root/'chain'/brief['path'];assert file_sha(path)==brief['sha256']
            ep=json.loads(gzip.open(path,'rt').read());eid=ep['episode_id'];isheld=eid>=splitpoint
            assert ep['evaluation'] is False and ep['end']=='GOAL' and ep['z']==ep['task_hp']/900
            raw=destination/'raw_sources'/origin;raw.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,raw/path.name)
            sources.append({'source':origin,'episode':eid,'path':str((raw/path.name).relative_to(destination)),
              'sha256':brief['sha256'],'split':'held' if isheld else 'train','positions':len(ep['records']),
              'model_id':ep['model_id'],'weight_hash':ep['weight_hash']})
            sa=a.initial;sb=b.initial
            for t,r in enumerate(ep['records']):
                assert r['step']==t and r['weight_hash']==ep['weight_hash']
                source_state=sa if origin=='map' else sb
                assert old.state_id(source_state)==r['state_id']
                assert old.observation_hash(old.observation.encode(source_state,t))==r['observation_hash']
                assert projected(sa,True)==projected(sb,False)
                ma=a.mask(sa);mb=b.mask(sb);assert np.array_equal(ma,mb)
                ids=np.flatnonzero(ma).tolist();assert ids==r['legal_action_ids'] and not ma[21:].any()
                assert abs(sum(r['pi'])-1)<1e-8
                assert np.allclose(np.array(r['root_N'])/sum(r['root_N']),r['pi'],rtol=0,atol=1e-12)
                pi=np.zeros(713,np.float32);pi[ids]=r['pi']
                xa=a.observation.encode(sa,t);xb=b.observation.encode(sb,t)
                decoded=unpack(xb);original=b.inner.observation.tensors(sb,t)
                assert all(np.array_equal(x,y) for x,y in zip(decoded,original))
                assert np.array_equal(xb,b.observation.encode(sb,t));roundtrips+=1
                assert np.isfinite(xa).all() and np.isfinite(xb).all() and np.isfinite(pi).all()
                idx=len(samples);xs['A'].append(xa);xs['B'].append(xb);masks.append(ma);pis.append(pi);zs.append(ep['z']);held.append(isheld)
                samples.append({'index':idx,'source':origin,'episode':eid,'step':t,'split':'held' if isheld else 'train',
                  'source_state_id':r['state_id'],'map_state_id':a.state_id(sa),'topology_state_id':b.state_id(sb),
                  'map_encoding_hash':a.observation_hash(xa),'topology_encoding_hash':b.observation_hash(xb),
                  'actions_sha256':action_sha,'legal_action_ids':ids,'original_pi':r['pi'],'z':ep['z'],
                  'selected_action_id':r['selected_action_id'],'source_model_id':ep['model_id'],'source_weight_hash':ep['weight_hash']})
                aid=r['selected_action_id'];sa=a.step(sa,aid)['state'];sb=b.step(sb,aid)['state']
                assert projected(sa,True)==projected(sb,False)
            assert a.terminal(sa)['task_hp']==b.terminal(sb)['task_hp']==ep['task_hp']
    held=np.asarray(held,bool);training=np.flatnonzero(~held);holdout=np.flatnonzero(held)
    assert len(sources)==52 and len(training)==801 and len(holdout)==139
    np.savez_compressed(destination/'paired.npz',A=np.stack(xs['A']),B=np.stack(xs['B']),
      masks=np.stack(masks),pi=np.stack(pis),z=np.asarray(zs,np.float32),training=training,held=holdout)
    for seed in [49001,49002,49003]:
        batches=training[np.random.RandomState(seed).randint(0,len(training),size=(56,64))]
        np.save(destination/f'batches-{seed}.npy',batches)
    write(destination/'samples.json',samples);write(destination/'sources.json',sources)
    n,r,h=b.inner.observation.tensors(b.initial,0);container=b.observation.encode(b.initial,0)
    write(destination/'tensor_example.json',{'meaning':'34x5x9 is only a serialization container, not physical map axes',
       'object_order':b.initial['graph']['order'],'fields':SLICES,'node_feature_names':b.inner.observation.channels,
       'nodes':n.tolist(),'relations':r.tolist(),'global':h.tolist(),'container':container.tolist(),
       'dtype':str(container.dtype),'shape':list(container.shape),'float_count':1530,'nonpadding_count':1456,
       'field_value_ranges':{key:[float(container.ravel()[lo:hi].min()),float(container.ravel()[lo:hi].max())] for key,(lo,hi) in SLICES.items()}})
    manifest={'training_episodes':44,'training_positions':len(training),'held_episodes':8,'held_positions':len(holdout),
      'roundtrip_checks':roundtrips,'actions_sha256':action_sha,'constant_value':float(np.asarray(zs,dtype=np.float64)[training].mean()),
      'constant_source':'training samples only; same constant on held set',
      'retained_duplicate_states':True,'source_state_replay_verified':True,
      'files':{str(p.relative_to(destination)):file_sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}}
    write(destination/'MANIFEST.json',manifest);return manifest
