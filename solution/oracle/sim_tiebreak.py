"""Vong 1c: BFS + tie-break F>R>L>B."""
import json, pathlib, collections
from collections import deque
BASE = pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
RANK={}  # (heading, action)->rank F=0,R=1,L=2,B=3
REL={('UP',0):0,('UP',3):1,('UP',2):2,('UP',1):3,
     ('DOWN',1):0,('DOWN',2):1,('DOWN',3):2,('DOWN',0):3,
     ('LEFT',2):0,('LEFT',0):1,('LEFT',1):2,('LEFT',3):3,
     ('RIGHT',3):0,('RIGHT',1):1,('RIGHT',0):2,('RIGHT',2):3}
def rc_to_action(a,b):
    return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((b[0]-a[0],b[1]-a[1]))
def build_adj(scene, legged=False):
    adj=collections.defaultdict(list)
    for e in scene['edges']:
        a,b=tuple(e['a']),tuple(e['b'])
        if e['status']=='closed': continue
        if e['stairs'] and not legged: continue
        if e['oneway_to'] is not None:
            t=tuple(e['oneway_to'])
            if t==b: adj[a].append((b,e))
            elif t==a: adj[b].append((a,e))
        else:
            adj[a].append((b,e)); adj[b].append((a,e))
    return adj
def pred_with_tiebreak(scene, legged=False):
    adj=build_adj(scene,legged)
    start=tuple(scene['robot']['rc']); head=scene['robot']['heading']
    m=scene['mission']
    lms=collections.defaultdict(list)
    for lm in scene['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    def resolve(typ,ref):
        if typ is None: return None
        if ref is not None: return tuple(ref['rc'])
        c=lms.get(typ,[])
        return c[0] if len(c)==1 else None
    via_rc=resolve(m['via'],m['via_ref']); goal_rc=resolve(m['goal'],m['goal_ref'])
    if goal_rc is None: return None,'ambig'
    wps=[w for w in [via_rc,goal_rc] if w is not None]
    cur=start; first=None
    for wp in wps:
        # BFS dist from cur
        ds={cur:0}; q=deque([cur])
        while q:
            u=q.popleft()
            for v,_ in adj.get(u,[]):
                if v not in ds: ds[v]=ds[u]+1; q.append(v)
        if wp not in ds: return None,'unreach'
        D=ds[wp]
        # reverse dist to wp
        radj=collections.defaultdict(list)
        for u,lst in adj.items():
            for v,_ in lst: radj[v].append(u)
        dg={wp:0}; q=deque([wp])
        while q:
            u=q.popleft()
            for v in radj.get(u,[]):
                if v not in dg: dg[v]=dg[u]+1; q.append(v)
        # candidates for first step of this leg (only first leg matters for output)
        if first is None:
            cands=[]
            for v,_ in adj.get(cur,[]):
                if ds.get(v,1e9)==1 and 1+dg.get(v,1e9)==D:
                    cands.append(rc_to_action(cur,v))
            if not cands: return None,'nocand'
            cands.sort(key=lambda a: REL[(head,a)])
            first=cands[0]
        cur=wp
    return first,'ok'

for split in ['train','validation']:
    scenes=json.loads((BASE/split/'scenes.json').read_text())
    labels=json.loads((BASE/split/'labels.json').read_text())
    # clean: single goal, no via? then broader: single goal any via-single?
    for name,filt in [
        ('clean-novía-single', lambda s: s['mission']['via'] is None and sum(1 for lm in s['landmarks'] if lm['type']==s['mission']['goal'])==1),
        ('single-goal-anyvia', lambda s: sum(1 for lm in s['landmarks'] if lm['type']==s['mission']['goal'])==1),
        ('all-resolvable', lambda s: True),
    ]:
        tot=hit=sk=0
        for i,s in enumerate(scenes):
            if not filt(s): continue
            # for all-resolvable skip ambig
            p,_=pred_with_tiebreak(s,legged=False)
            if p is None: sk+=1; continue
            tot+=1
            if p==labels[i*10+0]: hit+=1
        print(f'{split} {name}: acc={hit/max(1,tot):.4f} ({hit}/{tot}) skip={sk}')
