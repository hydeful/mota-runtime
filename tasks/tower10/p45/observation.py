"""Rules-derived 550-plane observation, from the frozen P42 architecture probe."""
import numpy as np
class ProposedCodec:
 def __init__(self,s):
  self.types=sorted({o['event'].get('id','') for o in s['objects'].values()}|{o['event'].get('id','') for o in s['rules']['templates'].values()});self.tids={v:i for i,v in enumerate(self.types)};self.traders=sorted(k[2:].replace(':','@')+'@A' for k,o in s['objects'].items() if o['trigger']=='trader');assert len(self.traders)==2
  self.statuses=['absent','active','hidden','resolved','moved'];self.per_floor=len(self.types)+19;self.channels=10*self.per_floor+20;self.s=s;self.walk=np.zeros((10,13,13),np.float32)
  self.stair=np.zeros((10,3,13,13),np.float32)
  for st in s['rules']['stairs']:
   f,cx,cy=st['position']
   if st['destination'][0]=='OUT_OF_SCOPE':
    assert not st['enabled'];continue
   df,dx,dy=st['destination']
   if df.startswith('MT') and 1<=int(df[2:])<=10:self.stair[int(f[2:])-1,:,cy,cx]=[int(df[2:])/10,dx/12,dy/12]
  for n in s['graph']['nodes'].values():self.mark(self.walk,n['position'],1)
  for es in s['graph']['out'].values():
   for e in es:
    for pos in e['path']:self.mark(self.walk,pos,1)
 def mark(self,a,pos,value):f,x,y=pos;a[int(f[2:])-1,y,x]=value
 def encode(self,s,t):
  x=np.zeros((self.channels,13,13),np.float32);K=len(self.types);pf=self.per_floor
  # Per-floor:34type +5status + walk/barrier/hero +4enemy +3door +excluded.
  assert pf==K+5+3+4+3+1+3
  for f in range(10):
   x[f*pf+K+5]=self.walk[f];x[f*pf+K+16:f*pf+K+19]=self.stair[f]
  for o in s['objects'].values():
   f,cx,cy=o['position'];i=(int(f[2:])-1)*pf;typ=o['event'].get('id','');assert typ in self.tids;st=o['status'];assert st in self.statuses
   x[i+self.tids[typ],cy,cx]=1;x[i+K+self.statuses.index(st),cy,cx]=1;x[i+K+15,cy,cx]=bool(o.get('excluded'))
   if o['trigger']=='battle':
    en=s['rules']['enemies'][typ]
    for k,(key,scale) in enumerate(zip(['hp','atk','def','money'],[10000,1000,1000,1000])):x[i+K+8+k,cy,cx]=en[key]/scale
   if o['trigger']=='openDoor':
    for k,key in enumerate(['yellowKey','blueKey','redKey']):x[i+K+12+k,cy,cx]=o['event'].get('doorInfo',{}).get('keys',{}).get(key,0)/10
  # Absence applies only where no typed object exists, including walls.
  for f in range(10):x[f*pf+K]=x[f*pf:f*pf+K].sum(0)==0
  for k in s['barriers']:
   _,f,cx,cy=k.split(':');x[(int(f[2:])-1)*pf+K+6,int(cy),int(cx)]=1
  h=s['hero'];x[(int(h['floorId'][2:])-1)*pf+K+7,h['y'],h['x']]=1
  tools=h['items'].get('tools',{});flags=s['flags'];gl=[h['hp']/6000,h['atk']/100,h['def']/100,h['money']/1000,*[tools.get(k,0)/100 for k in ['yellowKey','blueKey','redKey']],flags.get('times1',0)/6,float(bool(flags.get('8'))),float(bool(flags.get('10f机关'))),float('MT10@6@3@0' in flags.get('__aed__',[])),*[float(bool(flags.get(k))) for k in self.traders],t/346,float(t<30),float(s['task']['status']=='GOAL'),*[float(h['direction']==k) for k in ['up','right','down','left']]]
  assert len(gl)==20
  x[pf*10:]=np.array(gl,np.float32)[:,None,None];assert np.isfinite(x).all();return x

