"""Policy v3 (simple, grid-searched). Best per-robot cost+turn+tiebreak."""
import collections, heapq
def rc_to_action(a,b): return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((b[0]-a[0],b[1]-a[1]))
def rel(head,action):
    if action==head: return 'S'
    if (head,action) in [(0,3),(3,1),(1,2),(2,0)]: return 'R'
    if (head,action) in [(0,2),(2,1),(1,3),(3,0)]: return 'L'
    return 'B'
HEAD={'UP':0,'DOWN':1,'LEFT':2,'RIGHT':3}
def build_adj(sc,leg=False):
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
ZERO={'S':0,'R':0,'L':0,'B':0}
T1={'S':0,'R':1,'L':1,'B':3}
T2={'S':0,'R':1,'L':1,'B':5}
T_R5={'S':0,'R':2,'L':2,'B':20}
T_R8={'S':0,'R':0.5,'L':5,'B':10}
base=lambda e:1
COSTS={
 'base':base,
 'crowd1':lambda e:1+(1 if e['status']=='crowded' else 0),
 'crowd2':lambda e:1+(2 if e['status']=='crowded' else 0),
 'crowd5':lambda e:1+(5 if e['status']=='crowded' else 0),
 'cov0.3':lambda e:0.3 if e['status']=='covered' else 1,
 'cov0.5':lambda e:0.5 if e['status']=='covered' else 1,
 'covdisc':lambda e:0.5 if e['status']=='covered' else (1+(2 if e['status']=='crowded' else 0)),
 'c1c5':lambda e:(0.5 if e['status']=='covered' else 1)+(1 if e['status']=='crowded' else 0),
 'c2c5':lambda e:(0.5 if e['status']=='covered' else 1)+(2 if e['status']=='crowded' else 0),
}
# best per robot from grid search (cost, turn, tiebreak, leg)
PARAMS={
 0:('cov0.5','none','BLRS',False),
 1:('crowd2','none','SRLB',False),
 2:('crowd1','none','BLRS',False),
 3:('c1c5','none','RSLB',False),
 4:('crowd1','none','SRLB',True),
 5:('c1c5','tR5','SRLB',False),
 6:('crowd1','t2','LRSB',False),
 7:('crowd2','t2','BLSR',False),
 8:('crowd2','none','RSLB',False),
 9:(None,None,None,False),  # greedy
}
TURNS={'none':ZERO,'t1':T1,'t2':T2,'tR5':T_R5,'tR8':T_R8}
def cost_for(rid, sc=None):
    if rid==9: return None,False,None
    # R3 weather-conditional
    if rid==3 and sc is not None:
        if sc.get('weather')=='rain':
            return (lambda e:0.1 if e['status']=='covered' else 1),False,{'TC':TURNS['none'],'TB':'RSLB'}
        else:
            return (lambda e:1+(2 if e['status']=='crowded' else 0)),False,{'TC':TURNS['none'],'TB':'LRSB'}
    cn,tn,tb,leg=PARAMS[rid]
    return COSTS[cn],leg,{'TC':TURNS[tn],'TB':tb}
def dijkstra_first(adj,start,target,cost_fn,hd,tc,tb):
    tbr={d:i for i,d in enumerate(tb)}
    pq=[]
    for v,e in adj.get(start,[]):
        a=rc_to_action(start,v)
        heapq.heappush(pq,(cost_fn(e)+tc[rel(hd,a)], tbr[rel(hd,a)], v, a, a))
    vis={}
    while pq:
        c,r,u,din,fa=heapq.heappop(pq)
        if (u,din) in vis: continue
        vis[(u,din)]=1
        if u==target: return fa
        for v,e in adj.get(u,[]):
            a2=rc_to_action(u,v)
            if (v,a2) in vis: continue
            heapq.heappush(pq,(c+cost_fn(e)+tc[rel(din,a2)], r, v, a2, fa))
    return None
def path_cost(adj,start,target,cost_fn):
    import heapq as hq
    d={start:0}; pq=[(0,start)]
    while pq:
        dd,u=hq.heappop(pq)
        if dd!=d[u]: continue
        if u==target: return dd
        for v,e in adj.get(u,[]):
            nd=dd+cost_fn(e)
            if nd<d.get(v,1e9): d[v]=nd; hq.heappush(pq,(nd,v))
    return None
def bfs_dist(adj,start):
    import heapq as hq
    d={start:0}; pq=[(0,start)]
    while pq:
        dd,u=hq.heappop(pq)
        if dd!=d[u]: continue
        for v,e in adj.get(u,[]):
            if dd+1<d.get(v,1e9): d[v]=dd+1; hq.heappush(pq,(dd+1,v))
    return d
def predict_all(sc):
    start=tuple(sc['robot']['rc']); hd=HEAD[sc['robot']['heading']]; m=sc['mission']
    lms=collections.defaultdict(list)
    for lm in sc['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    def cands(typ,ref):
        if typ is None: return [None]
        if ref is not None: return [tuple(ref['rc'])]
        return lms.get(typ,[])
    vl=cands(m['via'],m['via_ref']); gl=cands(m['goal'],m['goal_ref'])
    if m['via'] is None: vl=[None]
    out={}
    for rid in range(9):
        fn,leg,pt=cost_for(rid,sc); tc=pt['TC']; tb=pt['TB']
        adj=build_adj(sc,leg)
        best=None;bk=None
        for vr in vl:
            for gr in gl:
                if gr is None: continue
                wps=[w for w in [vr,gr] if w is not None]
                cur=start;cd=hd;tot=0;first=None;ok=True
                for wp in wps:
                    fa=dijkstra_first(adj,cur,wp,fn,cd,tc,tb)
                    if fa is None and cur!=wp: ok=False;break
                    pc=path_cost(adj,cur,wp,fn)
                    if pc is None and cur!=wp: ok=False;break
                    tot+=pc or 0
                    if first is None: first=fa
                    cur=wp
                if not ok: continue
                k=(tot, ({d:i for i,d in enumerate(tb)}).get(rel(hd,first),99) if first is not None else 99)
                if bk is None or k<bk: bk=k;best=first
        out[rid]=best
    return out
def greedy9(sc):
    adj=build_adj(sc); rc=tuple(sc['robot']['rc']); head=sc['robot']['heading']; hd=HEAD[head]; m=sc['mission']
    nbrs=[v for v,e in adj.get(rc,[])]
    if not nbrs: return None
    lms=collections.defaultdict(list)
    for lm in sc['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    def cands(typ,ref):
        if typ is None: return []
        if ref is not None: return [tuple(ref['rc'])]
        return lms.get(typ,[])
    tl=cands(m['via'],m['via_ref']) if m['via'] is not None else cands(m['goal'],m['goal_ref'])
    tl=[t for t in tl if t is not None]
    if not tl: return None
    # v3: Euclidean distance + tie-break SBRL + crowded penalty
    tbr={'S':0,'B':1,'R':2,'L':3}
    emap={}
    for e in sc['edges']:
        a,b=tuple(e['a']),tuple(e['b'])
        if a==rc: emap[b]=e
        elif b==rc: emap[a]=e
    best=None;bk=None
    for nx in nbrs:
        d=min((nx[0]-t[0])**2+(nx[1]-t[1])**2 for t in tl)
        e=emap.get(nx)
        ec=2 if (e is not None and e['status']=='crowded') else 0
        a=rc_to_action(rc,nx)
        k=(d+ec,tbr[rel(hd,a)])
        if bk is None or k<bk: bk=k;best=a
    return best
