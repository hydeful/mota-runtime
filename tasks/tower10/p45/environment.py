"""Clean, versioned P41 T1 environment. Never calls legacy common.source()."""
import copy,copyreg,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from tasks.tower10.neural.common import Codec,canonical,digest,dynamic,state_id
from tasks.tower10.neural.vendor.kernel import prepare,_Tracked
from tasks.tower10.neural.vendor import topology_engine as reference
from tasks.tower10.engineering_upgrade.topology import InteractionSession,InteractionMask
from tasks.tower10.engineering_upgrade.fast import FastMask
from tasks.tower10.engineering_upgrade.actions import interaction_kind
from .observation import ProposedCodec

# Touched keys concern only construction-time deltas; next transition creates
# a fresh tracked map. Restore ordinary dicts, avoiding dict-subclass unpickle
# invoking __setitem__ before its touched attribute exists.
copyreg.pickle(_Tracked,lambda x:(dict,(dict(x),)))
DEPENDENCIES=['tasks/tower10/neural/__init__.py','tasks/tower10/neural/common.py','tasks/tower10/neural/vendor/__init__.py','tasks/tower10/neural/vendor/kernel.py','tasks/tower10/neural/vendor/path_index.py','tasks/tower10/neural/vendor/topology_engine.py','tasks/tower10/engineering_upgrade/__init__.py','tasks/tower10/engineering_upgrade/actions.py','tasks/tower10/engineering_upgrade/topology.py','tasks/tower10/engineering_upgrade/fast.py']
PINNED_INITIAL='80999d2e01da8253510c96bc2f52ba9b61fae19b74516b76bb58f25cd90a05e6'
PINNED_RULES='b78b162ef5a4e9ff353241eebc9f646cef7ed2e3b583f601ed3d1b81c83e806d'
PINNED_ENGINE='f602d445d832904eb54a030bc0bf8ebf4e9b9d76ea8f2860fccd34d8b07355d6'
PINNED_SOURCES={
 'tasks/tower10/neural/common.py':'921a99fa7ee7e0516ea794edb78760fdb15e6c2ad0d096e4eb7ea0dd820f4d1c',
 'tasks/tower10/neural/vendor/kernel.py':'e357d8785e283a9af2c5255488a9c8ede48ea72e70c9bde2410232d52fe3cc85',
 'tasks/tower10/neural/vendor/path_index.py':'48fb3d842f58c6191ec5d306ff50ba6ac85bd57b94c036a6f84a54e93ced7023',
 'tasks/tower10/neural/vendor/topology_engine.py':'f602d445d832904eb54a030bc0bf8ebf4e9b9d76ea8f2860fccd34d8b07355d6',
 'tasks/tower10/engineering_upgrade/__init__.py':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'tasks/tower10/engineering_upgrade/actions.py':'9d3587ef2a67361e399c2503d6a765ed2c72390b0997e8483d216e9b93dc1e4a',
 'tasks/tower10/engineering_upgrade/topology.py':'344aa8d561dc01ffc71d1fc9ee47438c6737eeada9f5ecd31b4f8bdd63756501',
 'tasks/tower10/engineering_upgrade/fast.py':'d08db70092878454afb571d65bca0bcbb248ef86f1550d631145a2a63a6c6bd7'
}

class Environment:
    def __init__(self,config=None,assets=None):
        here=Path(__file__).resolve().parent;assets=Path(assets) if assets else here/'assets'
        for name,h in PINNED_SOURCES.items():
            if hashlib.sha256((here.parents[2]/name).read_bytes()).hexdigest()!=h:raise ValueError('T1_PINNED_SOURCE_MISMATCH '+name)
        self.config=json.loads((here/'config.json').read_text()) if config is None else config
        self.audit_config()
        manifest=json.loads((assets/'SHA256.json').read_text())
        for name,h in manifest.items():
            if hashlib.sha256((assets/name).read_bytes()).hexdigest()!=h:raise ValueError('ASSET_HASH '+name)
        initial=json.loads((assets/'shared_initial_topology.json').read_text());self.identity=json.loads((assets/'shared_identity.json').read_text())
        if digest(initial)!=self.identity['initialSHA256'] or digest(initial['rules'])!=self.identity['rulesSHA256']:raise ValueError('GOLDEN_IDENTITY')
        if self.identity['initialSHA256']!=PINNED_INITIAL or self.identity['rulesSHA256']!=PINNED_RULES or self.identity['engineSHA256']!=PINNED_ENGINE:raise ValueError('UNTRUSTED_GOLDEN_OR_RULES')
        self.raw=prepare(initial);self.session=InteractionSession(self.raw);self.initial=self.raw.initial
        self.codec=Codec(self.initial)
        if len(self.codec.actions)!=713:raise ValueError('ACTION_VOCABULARY')
        self.mask=FastMask(self.codec,self.initial);self.reference_mask=InteractionMask(self.codec,self.initial)
        self.observation=ProposedCodec(self.initial)
        self.schema_description={'name':'P42-grid550-v1','shape':[550,13,13],'types':self.observation.types,'statuses':self.observation.statuses,'actions':self.codec.actions,'rules_sha':self.identity['rulesSHA256'],'event_read_set':self.audit_rules(),'encoder_source_sha256':hashlib.sha256((here/'observation.py').read_bytes()).hexdigest()}
        self.schema_hash=digest(self.schema_description)
        self.config_hash=digest({k:self.config[k] for k in ('algorithm','topology','search','reward','network','training')})
        source_files=sorted(here.glob('*.py'))
        pinned=[here.parents[2]/p for p in DEPENDENCIES if (here.parents[2]/p).is_file()]
        self.engine_hash=digest({str(p.relative_to(here) if p.is_relative_to(here) else p.relative_to(here.parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files+pinned})
        self.compatibility={'T':'T1.0','E':self.config['engineering'],'algorithm':self.config['algorithm'],'engine_hash':self.engine_hash,'golden_hash':self.identity['initialSHA256'],'rules_hash':self.identity['rulesSHA256'],'schema_hash':self.schema_hash,'actions_hash':digest(self.codec.actions),'search_hash':digest(self.config['search']),'reward_hash':digest(self.config['reward']),'architecture':f"residual-{self.config['network']['blocks']}x{self.config['network']['channels']}-P713-V1-tanh-BN-v1",'sample_schema':'P43-episode-v1','optimizer':'SGD-momentum0.9-lr0.01-weight_decay0'}
        self.compatibility['runtime']={'torch':str(torch.__version__),'numpy':np.__version__,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'deterministic':torch.are_deterministic_algorithms_enabled()}
        if self.observation.encode(self.initial,0).shape!=(550,13,13):raise ValueError('OBS_SHAPE')
    def audit_config(self):
        c=self.config;s=c['search'];n=c['network'];t=c['training']
        if t['min_new_positions']!=512:raise ValueError('TRAINING_TRIGGER_CONFIG')
        if s['pv_cache_entries']!=20000:raise ValueError('CACHE_CONFIG')
        if (n['blocks'],n['channels'],n['input'],n['policy'],n['value'])!=(2,32,[550,13,13],713,'tanh'):raise ValueError('UNSUPPORTED_NETWORK_CONFIG')
        if (s['new_simulations'],s['root_dirichlet_alpha'],s['root_noise_mix'],s['sample_first_interactions'],s['unvisited_Q'],s['gamma'],s['arbitrary_depth_cutoff'],s['plan_takeover'])!=(100,.3,.25,30,0,1,None,False):raise ValueError('UNSUPPORTED_SEARCH_CONFIG')
        if s['cpuct_kind']!='constant' or s['cpuct']!=1.25:raise ValueError('PUCT_CONFIG')
        if (t['replay_window'],t['logical_cohort'],t['batch'],t['exposures_per_new_position'],t['optimizer'],t['momentum'],t['lr'],t['manual_conv_linear_L2'],t['optimizer_weight_decay'])!=(8192,4,64,8,'SGD',.9,.01,.0001,0):raise ValueError('UNSUPPORTED_TRAINING_CONFIG')
        if c['reward']['goal_hp_scale']!=6000 or c['reward']['clip'] is not False:raise ValueError('REWARD_CONFIG')
        formulas={'H':'hp/5976','D':'(atk/37+def/47)/2','K':'(yellowKey/66+blueKey/5+redKey/1)/3','M':'money/662','S':'(H+D+K+M)/4','failure':'-1+0.9*S'}
        if any(c['reward'][k]!=v for k,v in formulas.items()) or c['reward']['bounds']!={'hp':5976,'atk':37,'def':47,'money':662,'yellowKey':66,'blueKey':5,'redKey':1}:raise ValueError('UNSUPPORTED_REWARD_FORMULA')
        if c['algorithm']!='A1.1' or c['engineering']!='E1.3' or c['topology']['version']!='T1.0' or c['topology']['source_freeze']!='d8012475702c948a4af0424930abddd6ac3d7129':raise ValueError('UNSUPPORTED_VERSION_CONFIG')
        if (torch.get_num_threads(),torch.get_num_interop_threads(),torch.are_deterministic_algorithms_enabled())!=(1,1,True):raise ValueError('FROZEN_RUNTIME_CONFIG')
    @staticmethod
    def observation_hash(x):return hashlib.sha256(np.asarray(x,dtype=np.float32).tobytes()).hexdigest()
    @staticmethod
    def dynamic(s):return {k:v for k,v in s.items() if k not in ('graph','rules')}
    @staticmethod
    def state_id(s):return state_id(s)
    def restore(self,d):
        if 'rules' in d or 'graph' in d:raise ValueError('DYNAMIC_INJECT_STATIC')
        return dict(d,graph=self.initial['graph'],rules=self.initial['rules'])
    def step(self,s,aid):
        if type(aid) is not int or not 0<=aid<713:raise ValueError('ACTION_ID')
        if s['task']['status']=='GOAL':raise ValueError('POST_GOAL_ACTION')
        return self.session.step(s,self.codec.actions[aid])
    def kind(self,s,aid):return interaction_kind(s,self.codec.actions[int(aid)])
    def diagnostics(self,s,ids):
        counts={k:0 for k in ('battle','door','pickup','shop_purchase','key_trade','one_time_npc','special_event')}
        growth=0
        for aid in ids:
            kind=self.kind(s,aid);counts[kind]+=1
            o=s['objects'][self.codec.actions[int(aid)]['target']]
            growth+=int(kind=='pickup' and o['event']['id'] in ('redGem','blueGem','redPotion','bluePotion','sword1','shield1'))
        return {'legal_count':len(ids),'legal_kinds':counts,'reachable_growth_supply':growth,'floor':int(s['hero']['floorId'][2:]),'resources':self.resource_score(s)}
    def audit_rules(self):
        flags=set();symbols=set();absent=set()
        def visit(c):
            op=c['op']
            if op=='flag':flags.add(c['name'])
            elif op=='unrecorded':symbols.add(c['symbol'])
            elif op=='absent':absent.add(c['slot'])
            elif op=='all':
                for v in c['args']:visit(v)
            elif op=='not':visit(c['arg'])
            elif op not in ('floor','constant'):raise ValueError('UNKNOWN_RULE_CONDITION '+op)
        for p in self.initial['rules']['eventPrograms']:visit(p['requires'])
        represented={'8','10f机关','10f战胜骷髅队长',*self.observation.traders}
        if flags-represented:raise ValueError('UNENCODED_RULE_FLAG '+repr(flags-represented))
        if symbols-{'MT10@6@3@0'}:raise ValueError('UNENCODED_AUTO_SYMBOL')
        return {'boolean_flags':sorted(flags),'unrecorded_symbols':sorted(symbols),'absent_objects':sorted(absent),'shop_counter':'times1','trader_flags':self.observation.traders,'excluded_fly_only':['__visited__','__leaveLoc__','fly','cannotMoveDirectly'],'bookkeeping_only_not_decision_predicates':['entities','events','serial','actionCount','hero.steps']}
    def resource_score(self,s):
        h=s['hero'];bag=h['items'].get('tools',{});b=self.config['reward']['bounds']
        vals={k:float(h[k]) for k in ('hp','atk','def','money')};vals.update({k:float(bag.get(k,0)) for k in ('yellowKey','blueKey','redKey')})
        for k,v in vals.items():
            if not np.isfinite(v) or v<0 or v>b[k]:raise ValueError('RESOURCE_BOUND '+repr((k,v,b[k])))
        H=vals['hp']/b['hp'];D=(vals['atk']/b['atk']+vals['def']/b['def'])/2;K=sum(vals[k]/b[k] for k in ('yellowKey','blueKey','redKey'))/3;M=vals['money']/b['money'];S=(H+D+K+M)/4
        if not 0<=S<=1:raise ValueError('RESOURCE_S_BOUND')
        return {'H':H,'D':D,'K':K,'M':M,'S':S,'failure_z':-1+.9*S,'actual':vals}
    def terminal(self,s,mask=None):
        score=self.resource_score(s)
        if s['task']['status']=='GOAL':
            if not s['flags'].get('10f战胜骷髅队长') or s['hero']['hp']<=0:raise ValueError('INCOMPLETE_GOAL')
            return {'end':'GOAL','task_hp':s['hero']['hp'],'z':s['hero']['hp']/6000,'resources':score,'proof':{'atomic_settled':True,'post_goal_actions':0}}
        if s['hero']['hp']<=0:raise ValueError('UNEXPECTED_DEATH_REQUIRES_RECEIPT')
        mask=self.mask(s) if mask is None else mask
        if mask.any():return None
        # Independently rerun the unoptimized reference T1 enumeration. A
        # disagreement is an engineering error, never a negative training label.
        reference_mask=self.reference_mask(s)
        if reference_mask.any():raise ValueError('INCOMPLETE_T1_ENUMERATION')
        legal=reference.legal_actions(s)
        supported=[a for a in legal if interaction_kind(s,a) is not None]
        if supported:raise ValueError('REFERENCE_EFFECTIVE_ACTION_MISSING')
        return {'end':'T1_NO_INTERACTION','task_hp':None,'z':score['failure_z'],'resources':score,'proof':{'T1_mask_complete':True,'reference_T1_empty':True,'reference_legal_count':len(legal),'unknown_effective_actions':0}}

def install_guard(run_root):
    """Reject legacy behavior directories and model/data files at file-open."""
    root=Path(__file__).resolve().parents[3];run_root=Path(run_root).resolve()
    here=Path(__file__).resolve().parent
    if run_root.is_relative_to(root) and not run_root.is_relative_to(here):raise ValueError('RUN_ROOT_INSIDE_LEGACY_REPO')
    allowed_sources={root/p for p in DEPENDENCIES}
    def source_allowed(path):
        if path in allowed_sources:return True
        if path.suffix=='.pyc' and path.parent.name=='__pycache__':
            return path.parent.parent/(path.name.split('.')[0]+'.py') in allowed_sources
        return False
    denied=[]
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        legacy=path.is_relative_to(root) and not (path.is_relative_to(here) or path.is_relative_to(run_root) or source_allowed(path))
        if legacy:denied.append(str(path));raise PermissionError('P43_OLD_BEHAVIOR_READ_BLOCKED '+str(path))
    sys.addaudithook(guard);return denied
