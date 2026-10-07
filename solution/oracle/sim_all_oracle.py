"""Tong hop oracle hien tai: macro train."""
import json, pathlib, collections, heapq, math
BASE = pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
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
def dijkstra_first(adj,start,target,cost_fn,heading):
    pq=[]
    for v,e in adj.get(start,[]):
        a=rc_to_action(start,v)
        heapq.heappush(pq,(cost_fn(e),REL[(heading,a)],v,a))
    vis={}
    while pq:
        c,r,u,fa=heapq.heappop(pq)
        if u in vis: continue
        vis[u]=(c,r,fa)
        if u==target: return fa
        for v,e in adj.get(u,[]):
            if v in vis: continue
            heapq.heappush(pq,(c+cost_fn(e),r,v,fa))
    return None
def predict_cost(scene,cost_fn,legged=False):
    adj=build_adj(scene,legged)
    start=tuple(scene['robot']['rc']); head=scene['robot']['heading']
    m=scene['mission']
    lms=collections.defaultdict(list)
    for lm in scene['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    def cands(typ,ref):
        if typ is None: return [None]
        if ref is not None: return [tuple(ref['rc'])]
        return lms.get(typ,[])
    via_list=cands(m['via'],m['via_ref']); goal_list=cands(m['goal'],m['goal_ref'])
    if m['via'] is None: via_list=[None]
    best=None; best_key=None
    for vr in via_list:
        for gr in goal_list:
            wps=[w for w in [vr,gr] if w is not None]
            cur=start; total=0; first=None; ok=True
            for wp in wps:
                fa=dijkstra_first(adj,cur,wp,cost_fn,head)
                if fa is None and cur!=wp: ok=False; break
                dist={cur:0}; pq=[(0,cur)]
                while pq:
                    d,u=heapq.heappop(pq)
                    if d!=dist[u]: continue
                    if u==wp: break
                    for v,e in adj.get(u,[]):
                        nd=d+cost_fn(e)
                        if nd<dist.get(v,1e9): dist[v]=nd; heapq.heappush(pq,(nd,v))
                if wp not in dist: ok=False; break
                total+=dist[wp]
                if first is None: first=fa
                cur=wp
            if not ok: continue
            key=(total, REL[(head,first)] if first is not None else 9)
            if best_key is None or key<best_key: best_key=key; best=first
    return best
def greedy_man(scene):
    rc=tuple(scene['robot']['rc']); head=scene['robot']['heading']
    nbrs=[]
    for e in scene['edges']:
        a,b=tuple(e['a']),tuple(e['b'])
        if e['status']=='closed': continue
        if e['stairs']: continue
        nxt=None
        if a==rc:
            if e['oneway_to'] is not None and tuple(e['oneway_to'])!=b: continue
            nxt=b
        elif b==rc:
            if e['oneway_to'] is not None and tuple(e['oneway_to'])!=a: continue
            nxt=a
        else: continue
        nbrs.append(nxt)
    m=scene['mission']
    lms=collections.defaultdict(list)
    for lm in scene['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    def cands(typ,ref):
        if typ is None: return [None]
        if ref is not None: return [tuple(ref['rc'])]
        return lms.get(typ,[])
    via_list=cands(m['via'],m['via_ref']); goal_list=cands(m['goal'],m['goal_ref'])
    tgt=[t for t in (via_list if m['via'] is not None else goal_list) if t is not None]
    best=None; bk=None
    for nxt in nbrs:
        d=min(abs(nxt[0]-t[0])+abs(nxt[1]-t[1]) for t in tgt)
        a=rc_to_action(rc,nxt)
        k=(d,REL[(head,a)])
        if bk is None or k<bk: bk=k; best=a
    return best

scenes=json.loads((BASE/'train/scenes.json').read_text())
labels=json.loads((BASE/'train/labels.json').read_text())
def cost_for(rid, scene):
    wthr=scene['weather']; urg=scene['mission']['urgent']; frag=scene['mission']['fragile']
    if rid==0: return (lambda e:1), False
    if rid==1: return (lambda e:1+(5 if e['status']=='crowded' else 0)), False
    if rid==2: return (lambda e:0.5 if e['status']=='covered' else 1), False
    if rid==3: return (lambda e:0.1 if e['status']=='covered' else 1) if wthr=='rain' else (lambda e:1+(3 if e['status']=='crowded' else 0)), False
    if rid==4: return (lambda e:1), True
    if rid==5: return (lambda e:1), False  # placeholder turn (chua crack)
    if rid==6: return (lambda e:1) if urg else (lambda e:1+(5 if e['status']=='crowded' else 0)), False
    if rid==7: return (lambda e:1) if not frag else (lambda e:1+(5 if e['status']=='crowded' else 0)), False  # placeholder fragile
    if rid==8: return (lambda e:1), False
    if rid==9: return None, False
for rid in range(10):
    tot=hit=0
    for i,s in enumerate(scenes):
        if rid==9: p=greedy_man(s)
        else:
            fn,leg=cost_for(rid,s); p=predict_cost(s,fn,leg)
        if p is None: continue
        tot+=1
        if p==labels[i*10+rid]: hit+=1
    print(f'R{rid}: {hit}/{tot}={hit/max(1,tot):.4f}')
