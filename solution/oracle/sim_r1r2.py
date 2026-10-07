"""Vong 2a: grid-search R1,R2,R4 + xu ly goal 2 ban."""
import json, pathlib, collections, heapq
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

def dijkstra_first(adj, start, target, cost_fn, heading):
    # cost, tie-break by relative rank of first step: implement via priority (cost, rank_first, path)
    # Dijkstra over (node) keeping best (cost, rank_first)
    import heapq
    best={}
    # pq: (cost, rank_first, node, first_action)
    # init neighbors
    pq=[]
    if start==target: return None
    for v,e in adj.get(start,[]):
        c=cost_fn(e); a=rc_to_action(start,v); r=REL[(heading,a)]
        heapq.heappush(pq,(c,r,v,a))
        # best[v] will be settled
    best_first={}
    visited_cost={}
    while pq:
        c,r,u,fa=heapq.heappop(pq)
        if u in visited_cost: continue
        # Since pq ordered by (c,r), first pop is optimal for that node under lexicographic order
        visited_cost[u]=(c,r,fa)
        if u==target: return fa
        for v,e in adj.get(u,[]):
            if v in visited_cost: continue
            nc=c+cost_fn(e)
            heapq.heappush(pq,(nc,r,v,fa))
    return None

def predict(scene, cost_fn, legged=False, goal_choice='mincost'):
    adj=build_adj(scene,legged)
    start=tuple(scene['robot']['rc']); head=scene['robot']['heading']
    m=scene['mission']
    lms=collections.defaultdict(list)
    for lm in scene['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    def cands(typ,ref):
        if typ is None: return [None]
        if ref is not None: return [tuple(ref['rc'])]
        return lms.get(typ,[])
    via_list=cands(m['via'],m['via_ref'])
    goal_list=cands(m['goal'],m['goal_ref'])
    if m['via'] is None: via_list=[None]
    if not goal_list: return None
    # enumerate combos (via x goal), pick min total cost (with tie-break)
    best=None; best_key=None; best_first=None
    for via_rc in via_list:
        for goal_rc in goal_list:
            wps=[w for w in [via_rc,goal_rc] if w is not None]
            cur=start; total=0; first=None; ok=True
            # compute leg costs + first
            for li,wp in enumerate(wps):
                fa=dijkstra_first(adj,cur,wp,cost_fn,head)
                if fa is None and cur!=wp: ok=False; break
                # compute cost of that leg (need dist)
                # recompute dist via dijkstra cost only
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
                if li==0 and first is None: first=fa
                # tie-break secondary: rank of first
                cur=wp
            if not ok: continue
            key=(total, REL[(head,first)] if first is not None else 9)
            if best_key is None or key<best_key:
                best_key=key; best_first=first
    return best_first

scenes=json.loads((BASE/'train/scenes.json').read_text())
labels=json.loads((BASE/'train/labels.json').read_text())

def eval_robot(rid, cost_fn, legged=False, subset=None):
    tot=hit=0
    for i,s in enumerate(scenes):
        if subset and not subset(s): continue
        p=predict(s,cost_fn,legged)
        if p is None: continue
        tot+=1
        if p==labels[i*10+rid]: hit+=1
    return hit/max(1,tot), hit, tot

# R1: avoid crowded
print('=== R1 grid w_crowded ===')
for w in [1,2,3,5,10,20]:
    fn=lambda e,w=w: 1 + (w if e['status']=='crowded' else 0)
    acc,h,t=eval_robot(1,fn)
    print(f' w={w}: {acc:.4f} {h}/{t}')
# R2: like covered: cost covered = 1-w? or 0.2?
print('=== R2 grid ===')
for cov in [0.1,0.2,0.3,0.5,0.7]:
    fn=lambda e,cov=cov: (cov if e['status']=='covered' else 1)
    acc,h,t=eval_robot(2,fn)
    print(f' cov_cost={cov}: {acc:.4f}')
for cov in [0.2,0.5]:
    for crowd in [2,5]:
        fn=lambda e,cov=cov,crowd=crowd: (cov if e['status']=='covered' else (crowd if e['status']=='crowded' else 1))
        acc,h,t=eval_robot(2,fn)
        print(f' cov={cov} crowd={crowd}: {acc:.4f}')
# R0 sanity with base cost
print('=== R0 sanity base ===')
acc,h,t=eval_robot(0,lambda e:1)
print(f' R0 base1: {acc:.4f}')
# R4 legged vs not
print('=== R4 ===')
for leg in [False,True]:
    acc,h,t=eval_robot(4,lambda e:1,legged=leg)
    print(f' legged={leg}: {acc:.4f}')
