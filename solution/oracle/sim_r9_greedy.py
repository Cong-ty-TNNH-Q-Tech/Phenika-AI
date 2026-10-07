"""Test R9 greedy: chon ke giam Euclid tới goal/via."""
import json, pathlib, collections, math
BASE = pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
def rc_to_action(a,b):
    return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((b[0]-a[0],b[1]-a[1]))
def valid_nbrs(scene, legged=False):
    rc=tuple(scene['robot']['rc'])
    out=[]
    for e in scene['edges']:
        a,b=tuple(e['a']),tuple(e['b'])
        if e['status']=='closed': continue
        if e['stairs'] and not legged: continue
        nxt=None
        if a==rc:
            if e['oneway_to'] is not None and tuple(e['oneway_to'])!=b: continue
            nxt=b
        elif b==rc:
            if e['oneway_to'] is not None and tuple(e['oneway_to'])!=a: continue
            nxt=a
        else: continue
        out.append((nxt,e))
    return out
scenes=json.loads((BASE/'train/scenes.json').read_text())
labels=json.loads((BASE/'train/labels.json').read_text())
REL={('UP',0):0,('UP',3):1,('UP',2):2,('UP',1):3,
     ('DOWN',1):0,('DOWN',2):1,('DOWN',3):2,('DOWN',0):3,
     ('LEFT',2):0,('LEFT',0):1,('LEFT',1):2,('LEFT',3):3,
     ('RIGHT',3):0,('RIGHT',1):1,('RIGHT',0):2,('RIGHT',2):3}
def greedy_pred(scene):
    nbrs=valid_nbrs(scene)
    if not nbrs: return None
    m=scene['mission']
    lms=collections.defaultdict(list)
    for lm in scene['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    # target: via if exists else goal; if dup no-ref, which? greedy picks closest Euclid? Try min over cands
    def cands(typ,ref):
        if typ is None: return [None]
        if ref is not None: return [tuple(ref['rc'])]
        return lms.get(typ,[])
    via_list=cands(m['via'],m['via_ref']); goal_list=cands(m['goal'],m['goal_ref'])
    tgt_list = via_list if m['via'] is not None else goal_list
    tgt_list=[t for t in tgt_list if t is not None]
    if not tgt_list: return None
    head=scene['robot']['heading']
    best=None; best_key=None
    for nxt,e in nbrs:
        # min Euclid to any candidate target
        d=min(math.hypot(nxt[0]-t[0],nxt[1]-t[1]) for t in tgt_list)
        a=rc_to_action(tuple(scene['robot']['rc']),nxt)
        key=(d,REL[(head,a)])
        if best_key is None or key<best_key: best_key=key; best=a
    return best

tot=hit=0
for i,s in enumerate(scenes):
    p=greedy_pred(s)
    if p is None: continue
    tot+=1
    if p==labels[i*10+9]: hit+=1
print(f'R9 greedy-Euclid (via-aware, min over dups): {hit}/{tot}={hit/max(1,tot):.4f}')
# subset single-goal no-via
tot=hit=0
for i,s in enumerate(scenes):
    if s['mission']['via'] is not None: continue
    if sum(1 for lm in s['landmarks'] if lm['type']==s['mission']['goal'])!=1: continue
    p=greedy_pred(s)
    tot+=1
    if p==labels[i*10+9]: hit+=1
print(f'R9 clean subset: {hit}/{tot}={hit/max(1,tot):.4f}')
# Try Manhattan instead?
def greedy_man(scene):
    nbrs=valid_nbrs(scene)
    m=scene['mission']
    lms=collections.defaultdict(list)
    for lm in scene['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    def cands(typ,ref):
        if typ is None: return [None]
        if ref is not None: return [tuple(ref['rc'])]
        return lms.get(typ,[])
    via_list=cands(m['via'],m['via_ref']); goal_list=cands(m['goal'],m['goal_ref'])
    tgt_list = via_list if m['via'] is not None else goal_list
    tgt_list=[t for t in tgt_list if t is not None]
    head=scene['robot']['heading']
    best=None; best_key=None
    for nxt,e in nbrs:
        d=min(abs(nxt[0]-t[0])+abs(nxt[1]-t[1]) for t in tgt_list)
        a=rc_to_action(tuple(scene['robot']['rc']),nxt)
        key=(d,REL[(head,a)])
        if best_key is None or key<best_key: best_key=key; best=a
    return best
tot=hit=0
for i,s in enumerate(scenes):
    p=greedy_man(s)
    if p is None: continue
    tot+=1
    if p==labels[i*10+9]: hit+=1
print(f'R9 greedy-Manhattan: {hit}/{tot}={hit/max(1,tot):.4f}')
