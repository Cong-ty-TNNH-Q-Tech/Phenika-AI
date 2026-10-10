"""Clean parameterized policy simulator. Tie-break = order of relative dirs (per-robot)."""
import json, pathlib, collections, heapq
def rc_to_action(a,b): return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((b[0]-a[0],b[1]-a[1]))
def rel_dir(head, action):
    if action==head: return 'S'
    if (head,action) in [(0,3),(3,1),(1,2),(2,0)]: return 'R'
    if (head,action) in [(0,2),(2,1),(1,3),(3,0)]: return 'L'
    return 'B'
def build_adj(sc, leg=False):
    adj=collections.defaultdict(list)
    for e in sc['edges']:
        a,b=tuple(e['a']),tuple(e['b'])
        if e['status']=='closed': continue
        if e['stairs'] and not leg: continue
        if e['oneway_to'] is not None:
            t=tuple(e['oneway_to'])
            if t==b: adj[a].append((b,e))
            elif t==a: adj[b].append((a,e))
        else: adj[a].append((b,e)); adj[b].append((a,e))
    return adj
HEAD2DIR={'UP':0,'DOWN':1,'LEFT':2,'RIGHT':3}
def dijkstra_first(adj, start, target, cost_fn, head_dir, tb_rank, turn_cost=None):
    """Return first action of min-cost path start->target.
    tb_rank: dict rel_dir->rank (lower=better). turn_cost: dict S/R/L/B->cost (optional)."""
    tc=turn_cost or {'S':0,'R':0,'L':0,'B':0}
    pq=[]
    for v,e in adj.get(start,[]):
        a=rc_to_action(start,v)
        c=cost_fn(e)+tc[rel_dir(head_dir,a)]
        heapq.heappush(pq,(c, tb_rank[rel_dir(head_dir,a)], v, a, a))
    vis={}
    while pq:
        c,r,u,din,fa=heapq.heappop(pq)
        if (u,din) in vis: continue
        vis[(u,din)]=1
        if u==target: return fa
        for v,e in adj.get(u,[]):
            a2=rc_to_action(u,v)
            if (v,a2) in vis: continue
            heapq.heappush(pq,(c+cost_fn(e)+tc[rel_dir(din,a2)], r, v, a2, fa))
    return None
def path_cost(adj, start, target, cost_fn, head_dir, turn_cost=None):
    tc=turn_cost or {'S':0,'R':0,'L':0,'B':0}
    dist={start:0}; pq=[(0,start)]
    while pq:
        d,u=heapq.heappop(pq)
        if d!=dist[u]: continue
        if u==target: return d
        for v,e in adj.get(u,[]):
            nd=d+cost_fn(e)
            if nd<dist.get(v,1e9): dist[v]=nd; heapq.heappush(pq,(nd,v))
    return None
def predict(sc, cost_fn, leg, tb_order, turn_cost=None):
    """tb_order: string like 'SRLB' or 'SRLB' -> rank. Returns first action."""
    adj=build_adj(sc,leg); start=tuple(sc['robot']['rc']); head=sc['robot']['heading']; hd=HEAD2DIR[head]
    tb_rank={d:i for i,d in enumerate(tb_order)}
    m=sc['mission']; lms=collections.defaultdict(list)
    for lm in sc['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    def cands(typ,ref):
        if typ is None: return [None]
        if ref is not None: return [tuple(ref['rc'])]
        return lms.get(typ,[])
    vl=cands(m['via'],m['via_ref']); gl=cands(m['goal'],m['goal_ref'])
    if m['via'] is None: vl=[None]
    best=None; bk=None
    for vr in vl:
        for gr in gl:
            if gr is None: continue
            wps=[w for w in [vr,gr] if w is not None]
            cur=start; cur_dir=hd; total=0; first=None; ok=True
            for wp in wps:
                fa=dijkstra_first(adj,cur,wp,cost_fn,cur_dir,tb_rank,turn_cost)
                if fa is None and cur!=wp: ok=False; break
                pc=path_cost(adj,cur,wp,cost_fn,cur_dir,turn_cost)
                if pc is None and cur!=wp: ok=False; break
                total+=pc or 0
                if first is None: first=fa
                cur=wp
            if not ok: continue
            k=(total, tb_rank[rel_dir(hd,first)] if first is not None else 99)
            if bk is None or k<bk: bk=k; best=first
    return best
def eval_robot(scenes, labels, rid, cost_fn, leg, tb_order, turn_cost=None):
    hit=tot=0
    for i,s in enumerate(scenes):
        p=predict(s,cost_fn,leg,tb_order,turn_cost)
        if p is None: continue
        tot+=1
        if p==labels[i*10+rid]: hit+=1
    return hit/max(1,tot),hit,tot
