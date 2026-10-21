"""P22 pure topology transition kernel. No game-map, Simulator or mapping imports.

The graph is a directed corridor multigraph compiled once by bootstrap. Empty
space is routing evidence, never an action target. Slot IDs remain compatible
with P21; entities have independent identities and lifecycles.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from collections import deque

DIRECTIONS={(0,-1):'up',(1,0):'right',(0,1):'down',(-1,0):'left'}

def digest(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def slot(p): return 'o:'+':'.join(map(str,p))
def vertex(p): return 'v:'+':'.join(map(str,p))
def position(s): return [s['hero']['floorId'],s['hero']['x'],s['hero']['y']]
def active(s,k):
    o=s['objects'].get(k)
    return o if o and o['status']=='active' else None

def damage(s,enemy):
    e=s['rules']['enemies'][enemy];h=s['hero']
    if h['atk']<=e['def']:return None
    return (math.ceil(e['hp']/(h['atk']-e['def']))-1)*max(0,e['atk']-h['def'])

def paths(s):
    """Current next-object frontiers, computed solely from corridor topology."""
    start=vertex(position(s));queue=deque([(start,[])]);seen={start};found={}
    while queue:
        p,trail=queue.popleft()
        for edge in s['graph']['out'].get(p,[]):
            q=edge['target'];qs=s['graph']['nodes'][q]['position'];k=slot(qs)
            if k in s['barriers']:continue
            path=trail+edge['path'][1:]
            obj=active(s,k)
            if obj:
                if q!=start:found.setdefault(k,path)
                continue
            if q not in seen:seen.add(q);queue.append((q,path))
    return found

def _stair_effective(s,z):
    o=s['objects'].get(slot(z['position']))
    return o['status']=='active' if o else z['enabled']

def _direct_reachable(s,fid,source):
    src_obj=active(s,slot([fid,*source]))
    if src_obj and src_obj['trigger']!='changeFloor':return set()
    start=vertex([fid,*source]);todo=[start];seen={start};stairs={vertex(z['position']) for z in s['rules']['stairs'] if _stair_effective(s,z)}
    while todo:
        p=todo.pop()
        for e in s['graph']['out'].get(p,[]):
            q=e['target'];pos=s['graph']['nodes'][q]['position'];k=slot(pos)
            if pos[0]!=fid or q in stairs or any(z[0]!=fid for z in e['path']) or active(s,k) or k in s['barriers']:continue
            if q not in seen:seen.add(q);todo.append(q)
    return seen

def fly_refusal(s,fid):
    if s['task']['status']=='GOAL':return 'AFTER_GOAL'
    cfg=s['rules']['fly'];h=s['hero'];f=s['flags'];current=h['floorId'];floors=cfg['floors']
    if fid not in floors:return 'OUT_OF_TASK_RANGE'
    if not h['items'].get('constants',{}).get('fly'):return 'NO_FLY_ITEM'
    if not floors[current]['canFlyFrom'] or not floors[fid]['canFlyTo']:return 'FLY_FLOOR_FORBIDDEN'
    if not any(f.get('__visited__',{}).get('MT'+str(n)) for n in range(int(fid[2:]),11)):return 'FLOOR_UNVISITED'
    src=[h['x'],h['y']];near=any(z['position'][0]==current and abs(z['position'][1]-src[0])+abs(z['position'][2]-src[1])<=1 for z in s['rules']['stairs'] if _stair_effective(s,z))
    direct=_direct_reachable(s,current,src)
    if cfg['nearStair'] and not near:
        extra=f.get('额外功能开关') and cfg['directEnabled'] and not floors[current]['cannotMoveDirectly'] and not f.get('cannotMoveDirectly')
        stair_neighbor=any(vertex([current,z['position'][1]+dx,z['position'][2]+dy]) in direct for z in s['rules']['stairs'] if z['position'][0]==current for dx,dy in ((0,-1),(1,0),(0,1),(-1,0)))
        if not extra or not stair_neighbor:return 'FLY_MENU_UNAVAILABLE'
    if not f.get('fly'):
        def connection(floor,source,dest,plugin=True):
            return source==dest if plugin and source==dest else vertex([floor,*dest]) in _direct_reachable(s,floor,source) and not any(z['position']==[floor,*dest] for z in s['rules']['stairs'] if _stair_effective(s,z))
        if current==fid:
            if not connection(current,src,floors[current]['downFloor'],False):return 'NATIVE_FLOOR_DISCONNECTED'
        else:
            a,b=int(current[2:]),int(fid[2:]);d=1 if b>a else -1
            if not connection(current,src,floors[current]['upFloor' if d>0 else 'downFloor']):return 'NATIVE_FLOOR_DISCONNECTED'
            for n in range(a+d,b,d):
                floor='MT'+str(n);fd=floors[floor]
                if not fd['isHide'] and not connection(floor,fd['upFloor'],fd['downFloor']):return 'NATIVE_FLOOR_DISCONNECTED'
    return None

def _fly(s,fid):
    h=s['hero'];current=h['floorId'];loc={'x':h['x'],'y':h['y'],'direction':h['direction']}
    stair='downFloor' if int(current[2:])<=int(fid[2:]) else 'upFloor';dest=s['rules']['fly']['floors'][fid][stair]
    s['flags'].setdefault('__leaveLoc__',{})[current]=loc;h.update(floorId=fid,x=dest[0],y=dest[1]);s['flags']['对话能量']=0
    if not any(s['flags'].get('__visited__',{}).get('MT'+str(n)) for n in range(int(fid[2:]),11)):s['flags'].setdefault('__visited__',{})[fid]=True
    _automatic(s);s['actionCount']+=1

def refusal(s,k,choices=None):
    if s['task']['status']=='GOAL':return 'AFTER_GOAL'
    o=active(s,k)
    if not o:return 'OBJECT_NOT_ACTIVE'
    if o.get('excluded'):return 'OUT_OF_SCOPE_BY_USER'
    if k not in paths(s):return 'NOT_NEXT_OBJECT'
    t=o['trigger'];e=o['event']
    if t=='battle':
        d=damage(s,e['id'])
        if d is None:return 'CANNOT_BATTLE'
        if d>=s['hero']['hp']:return 'WOULD_DIE'
    if t=='openDoor':
        for key,n in (e.get('doorInfo',{}).get('keys',{})).items():
            if s['hero']['items'].get('tools',{}).get(key,0)<n:return 'INSUFFICIENT_KEY'
        if e['id'] in ('specialDoor','steelDoor'):return 'MECHANISM_DOOR'
    if k=='o:MT4:6:1':
        cs=choices if choices is not None else [3]
        if not isinstance(cs,list) or not cs or cs[-1]!=3 or any(c not in (0,1,2) for c in cs[:-1]):return 'INVALID_MENU'
        money=s['hero']['money'];n=s['flags'].get('times1',0)
        for c in cs[:-1]:
            cost=20+10*(n+1)*n
            if money<cost:return 'INSUFFICIENT_MONEY'
            money-=cost;n+=1
    elif t=='trader' and not s['flags'].get(k[2:].replace(':','@')+'@A'):
        cs=choices if choices is not None else [1]
        if cs not in ([0],[1]):return 'INVALID_MENU'
        if cs==[0] and s['hero']['money']<50:return 'INSUFFICIENT_MONEY'
    elif t=='oldman' and o['position'][0]=='MT3':
        if choices not in (None,[0]):return 'INVALID_MENU'
    elif choices:return 'UNEXPECTED_MENU'
    return None

def legal_actions(s):
    if s['task']['status']=='GOAL':return []
    out=[]
    for k in sorted(paths(s)):
        o=active(s,k)
        cs=[[3],[0,3],[1,3],[2,3]] if k=='o:MT4:6:1' else [[0],[1]] if o['trigger']=='trader' and not s['flags'].get(k[2:].replace(':','@')+'@A') else [[0]] if o['trigger']=='oldman' and o['position'][0]=='MT3' else [None]
        for choice in cs:
            a={'target':k}
            if choice is not None:a['choices']=choice
            if refusal(s,k,choice) is None:out.append(a)
    out.extend({'target':'hero','fly':fid} for fid in sorted(s['rules']['fly']['floors']) if fly_refusal(s,fid) is None)
    return out

def _membership(s,k,enabled):
    _,fid,x,y=k.split(':');idx=int(x)+int(y)*s['rules']['widths'][fid]
    pair=s['flags'].setdefault('__disabled__',{}).setdefault(fid,[[],[]])
    pair[0]=sorted(set(pair[0])-{idx}) if enabled else sorted(set(pair[0])|{idx})
    pair[1]=sorted(set(pair[1])|{idx}) if enabled else sorted(set(pair[1])-{idx})

def _remove(s,k):
    o=s['objects'].get(k)
    if o and o['status'] in ('active','hidden'):
        o['status']='resolved';s['entities'][o['entity']]['status']='resolved'
        s['entities'][o['entity']]['lastSlot']=k
    s['barriers'].pop(k,None)
    _membership(s,k,False)

def _spawn(s,k,template,eid=None):
    old=s['objects'].get(k)
    origin_exists=bool(old and old['status'] in ('active','hidden')) or k in s['barriers']
    membership_before=copy.deepcopy(s['flags'].get('__disabled__',{})) if origin_exists else None
    if old and old['status'] in ('active','hidden'):_remove(s,k)
    if origin_exists:s['flags']['__disabled__']=membership_before
    eid=eid or 'born:'+k+':'+str(s['serial']);s['serial']+=1
    o=copy.deepcopy(template);o.update(id=k,status='active',entity=eid)
    s['objects'][k]=o;s['entities'].setdefault(eid,{}).update(status='active',slot=k,enemy=o['event']['id'])
    s['barriers'].pop(k,None)
    if not origin_exists:_membership(s,k,True)

def _ambush(s):
    rule=s['rules']['ambush'];f=s['flags']
    for p in rule['empty']:_remove(s,slot(p))
    for move in rule['moves']:
        src=slot(move['source']);dst=slot(move['destination']);old=active(s,src)
        eid=old['entity'] if old and old['event']['id']==move['enemy'] else move['futureEntity']
        if old and old['event']['id']==move['enemy']:
            old['status']='moved';s['entities'][eid]['status']='moving'
            s['entities'][move['futureEntity']].update(status='alias',aliasOf=eid)
            _membership(s,src,False)
        elif move['enemy']!='skeletonCaptain':
            p=move['source'];s['flags'].setdefault('__bgv__',{}).setdefault(p[0],[]).append([p[1],p[2],0])
        _spawn(s,dst,s['rules']['templates'][str(move['tile'])]|{'position':move['destination']},eid)
    for p in rule['doors']:_spawn(s,slot(p),s['rules']['templates']['85']|{'position':p},'door:ambush:'+slot(p))
    f['MT10@6@5@A']=True;f['10f机关']=True;_remove(s,'o:MT10:6:5');s['events']['MT10.ambush']='settled'


def evaluate(s,c):
    op=c['op']
    if op=='all':return all(evaluate(s,a) for a in c['args'])
    if op=='not':return not evaluate(s,c['arg'])
    if op=='flag':return bool(s['flags'].get(c['name']))
    if op=='absent':return not active(s,c['slot']) and c['slot'] not in s['barriers']
    if op=='floor':return s['hero']['floorId']==c['floorId']
    if op=='unrecorded':return c['symbol'] not in s['flags'].get('__aed__',[])
    if op=='constant':return c['value']
    raise ValueError('UNKNOWN_CONDITION '+op)

def _programs(s,trigger,k=None):
    # Freeze the matches before applying effects: mutually exclusive branches
    # (e.g. first/second bound guard) cannot cascade within the same trigger.
    matches=[p for p in s['rules']['eventPrograms'] if p['trigger']['op']==trigger and (p['trigger'].get('object') in (None,k)) and evaluate(s,p['requires'])]
    for p in matches:
        for e in p['effects']:
            op=e['op']
            if op=='remove':_remove(s,e['object'])
            elif op=='flag_set':s['flags'][e['name']]=e['value']
            elif op=='flag_add':s['flags'][e['name']]=s['flags'].get(e['name'],0)+e['value']
            elif op=='auto_record':s['flags'].setdefault('__aed__',[]).append(e['symbol'])
            elif op=='ambush':_ambush(s)
            elif op=='boss_reward':_boss(s)
            else:raise ValueError('UNKNOWN_EFFECT '+op)
        s['events'][p['id']]='settled'

def _boss(s):
    for r in s['rules']['rewards']:_spawn(s,slot(r['position']),s['rules']['templates'][str(r['tile'])]|{'position':r['position']},r['entity'])
    for q in [('MT10',4,4),('MT10',6,7),('MT10',8,4)]:_remove(s,slot(q))
    _spawn(s,'o:MT10:6:9',s['rules']['bossThiefEvent'],'future:MT10.thief_after_goal')
    for q in [('MT10',6,11)]:
        z=s['objects'].get(slot(q))
        if z and z['status']=='hidden':
            z['status']='active';s['entities'][z['entity']]['status']='active';_membership(s,slot(q),True)
    s['flags']['10f战胜骷髅队长']=True;s['task']['status']='GOAL'

def _automatic(s):
    _programs(s,'automatic')

def _walk(s,path,no_pass):
    route=[];h=s['hero'];f=s['flags'];prev=position(s)
    for i,q in enumerate(path):
        if q[0]!=prev[0]:
            f.setdefault('__leaveLoc__',{})[prev[0]]={'x':prev[1],'y':prev[2],'direction':h['direction']}
            h.update(floorId=q[0],x=q[1],y=q[2]);f['对话能量']=0
            if not any(f.get('__visited__',{}).get('MT'+str(n)) for n in range(int(q[0][2:]),11)):f.setdefault('__visited__',{})[q[0]]=True
            prev=q;continue
        dx,dy=q[1]-prev[1],q[2]-prev[2]
        if not (dx or dy):continue
        direction=DIRECTIONS[(dx,dy)];route.append(direction);h['direction']=direction
        if not (no_pass and i==len(path)-1):
            h.update(x=q[1],y=q[2]);h['steps']=h.get('steps',0)+1;f['对话能量']=f.get('对话能量',0)-1
            _automatic(s)
        prev=q
    return route

def _delta(before,after):
    def changes(key):
        a,b=before[key],after[key]
        return {k:{'before':copy.deepcopy(a.get(k)),'after':copy.deepcopy(b.get(k))} for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
    return {key:changes(key) for key in ('hero','flags','objects','entities','barriers','events','task')}

def _valid_action(action):
    if not isinstance(action,dict) or set(action)-{'target','choices','fly'}:return False
    if not isinstance(action.get('target'),str):return False
    choices=action.get('choices')
    if choices is not None and (not isinstance(choices,list) or any(type(c) is not int for c in choices)):return False
    if 'fly' in action and (action['target']!='hero' or not isinstance(action['fly'],str) or choices):return False
    return True

def step(state,action):
    """Transactional execute. Rejection returns an unchanged independent copy."""
    valid=_valid_action(action)
    k=action.get('target') if valid else None;cs=action.get('choices') if valid else None
    reason='MALFORMED_ACTION' if not valid else fly_refusal(state,action['fly']) if 'fly' in action else refusal(state,k,cs)
    if reason:return {'state':copy.deepcopy(state),'result':{'accepted':False,'reason':reason,'action':copy.deepcopy(action)},'delta':{},'routeSegment':[]}
    if 'fly' in action:
        s=copy.deepcopy(state);_fly(s,action['fly']);return {'state':s,'result':{'accepted':True,'action':copy.deepcopy(action),'goal':False},'delta':_delta(state,s),'routeSegment':['fly:'+action['fly']]}
    s=copy.deepcopy(state);o=active(s,k);p=o['position'];t=o['trigger'];e=o['event'];h=s['hero'];f=s['flags']
    path=paths(s)[k];route=_walk(s,path,e.get('noPass',False) and t!='battle');effects=[]
    if t=='battle':
        d=damage(s,e['id']);h['hp']-=d;h['money']+=s['rules']['enemies'][e['id']]['money'];h['experience']=None
        f['hatred']=f.get('hatred',0)+s['rules']['values'].get('hatred',0);_remove(s,k);f['对话能量']=0
        _programs(s,'after_battle',k)
    elif t=='getItem':
        item=e['id'];f.setdefault('__itemHint__',[]);r=s['rules'];cls=r['items'][item]['cls']
        if item in ('redGem','blueGem','redPotion','bluePotion'):
            stat={'redGem':'atk','blueGem':'def','redPotion':'hp','bluePotion':'hp'}[item];h[stat]+=r['values'][item]*r['ratios'][p[0]]
        elif item in ('sword1','shield1'):
            h['atk' if item=='sword1' else 'def']+=10;f['nowWeapon' if item=='sword1' else 'nowShield']=item
        else:
            bag=h['items'].setdefault(cls,{});bag[item]=min(1,bag.get(item,0)+1) if cls=='constants' else bag.get(item,0)+1
        _remove(s,k)
        _programs(s,'after_pickup',k)
    elif t=='openDoor':
        bag=h['items'].setdefault('tools',{})
        for key,n in e['doorInfo'].get('keys',{}).items():
            bag[key]-=n
            if bag[key]<=0:del bag[key]
        _remove(s,k)
    elif k=='o:MT4:6:1':
        f['ratio']=1
        cs=cs or [3]
        for c in cs:
            n=f.get('times1',0);f['money1']=20+10*(n+1)*n
            if c==3:break
            stat=['hp','atk','def'][c];amount=100*(n+1) if c==0 else (2 if c==1 else 4)
            h['money']-=f['money1'];h[stat]+=amount;f['times1']=n+1
            f['1区购买'+stat]=f.get('1区购买'+stat,0)+amount;f['1区购买'+stat+'次数']=f.get('1区购买'+stat+'次数',0)+1
        route+=['choices:'+str(c) for c in cs]
    elif t in ('oldman','trader'):
        n=int(p[0][2:]);bought=k[2:].replace(':','@')+'@A'
        if t=='trader' and not f.get(bought):
            cs=cs or [1];f['money']=50;f['alt']=0
            if cs==[0]:
                h['money']-=50;item,num=('blueKey',1) if n==6 else ('yellowKey',5)
                bag=h['items'].setdefault('tools',{});bag[item]=bag.get(item,0)+num;f[bought]=1
            route+=['choices:'+str(c) for c in cs]
        else:
            if n==3:
                h['items'].setdefault('constants',{})['book']=1;route+=['choices:0']
            f['有选择的对话']=n==3;f['对话'+str(n)+('02' if t=='trader' else '')]=True
            _remove(s,k);f['对话能量']=2
    elif k=='o:MT10:6:5':_programs(s,'step_into',k)
    elif k=='o:MT10:6:2':
        _remove(s,k)
    elif k=='o:MT10:6:9':pass
    else:raise ValueError('UNSUPPORTED_EVENT '+str(k))
    _automatic(s);s['actionCount']+=1
    delta=_delta(state,s)
    return {'state':s,'result':{'accepted':True,'action':copy.deepcopy(action),'goal':s['task']['status']=='GOAL'},'delta':delta,'routeSegment':route}

execute=step

def compile_actions(initial,actions):
    state=copy.deepcopy(initial);route=[];records=[]
    for i,a in enumerate(actions):
        r=step(state,a)
        if not r['result']['accepted']:raise ValueError({'index':i,'action':a,'result':r['result'],'stateHash':digest(state)})
        state=r['state'];route.extend(r['routeSegment']);records.append(r['result'])
    return {'state':state,'route':route,'records':records}
