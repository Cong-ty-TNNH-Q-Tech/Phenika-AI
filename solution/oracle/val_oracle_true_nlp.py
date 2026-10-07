import json, pathlib, collections, heapq
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
REL={('UP',0):0,('UP',3):1,('UP',2):2,('UP',1):3,('DOWN',1):0,('DOWN',2):1,('DOWN',3):2,('DOWN',0):3,('LEFT',2):0,('LEFT',0):1,('LEFT',1):2,('LEFT',3):3,('RIGHT',3):0,('RIGHT',1):1,('RIGHT',0):2,('RIGHT',2):3}
def rc_to_action(a,b):
    return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((b[0]-a[0],b[1]-a[1]))
def build_adj(sc,leg):
    adj=collections.defaultdict(list)
    for e in sc['edges']:
        a,b=tuple(e['a']),tuple(e['b'])
        if e['status']=='closed':
            continue
        if e['stairs'] and not leg:
            continue
        if e['oneway_to'] is not None:
            t=tuple(e['oneway_to'])
            if t==b:
                adj[a].append((b,e))
            elif t==a:
                adj[b].append((a,e))
        else:
            adj[a].append((b,e))
            adj[b].append((a,e))
    return adj
def dfirst(adj,s,t,cf,hd):
    pq=[]
    for v,e in adj.get(s,[]):
        a=rc_to_action(s,v)
        heapq.heappush(pq,(cf(e),REL[(hd,a)],v,a))
    vis={}
    while pq:
        c,r,u,fa=heapq.heappop(pq)
        if u in vis:
            continue
        vis[u]=1
        if u==t:
            return fa
        for v,e in adj.get(u,[]):
            if v in vis:
                continue
            heapq.heappush(pq,(c+cf(e),r,v,fa))
    return None
def pred_cost(sc,cf,leg):
    adj=build_adj(sc,leg)
    start=tuple(sc['robot']['rc'])
    head=sc['robot']['heading']
    m=sc['mission']
    lms=collections.defaultdict(list)
    for lm in sc['landmarks']:
        lms[lm['type']].append(tuple(lm['rc']))
    def cands(typ,ref):
        if typ is None:
            return [None]
        if ref is not None:
            return [tuple(ref['rc'])]
        return lms.get(typ,[])
    vl=cands(m['via'],m['via_ref'])
    gl=cands(m['goal'],m['goal_ref'])
    if m['via'] is None:
        vl=[None]
    best=None
    bk=None
    for vr in vl:
        for gr in gl:
            if gr is None:
                continue
            wps=[w for w in [vr,gr] if w is not None]
            cur=start
            tot=0
            first=None
            ok=True
            for wp in wps:
                fa=dfirst(adj,cur,wp,cf,head)
                if fa is None and cur!=wp:
                    ok=False
                    break
                dist={cur:0}
                pq=[(0,cur)]
                while pq:
                    d,u=heapq.heappop(pq)
                    if d!=dist[u]:
                        continue
                    if u==wp:
                        break
                    for v,e in adj.get(u,[]):
                        nd=d+cf(e)
                        if nd<dist.get(v,1e9):
                            dist[v]=nd
                            heapq.heappush(pq,(nd,v))
                if wp not in dist:
                    ok=False
                    break
                tot+=dist[wp]
                if first is None:
                    first=fa
                cur=wp
            if not ok:
                continue
            k=(tot,REL[(head,first)] if first is not None else 9)
            if bk is None or k<bk:
                bk=k
                best=first
    return best
def greedy(sc):
    rc=tuple(sc['robot']['rc'])
    head=sc['robot']['heading']
    nbrs=[]
    for e in sc['edges']:
        a,b=tuple(e['a']),tuple(e['b'])
        if e['status']=='closed' or e['stairs']:
            continue
        nxt=None
        if a==rc:
            if e['oneway_to'] is not None and tuple(e['oneway_to'])!=b:
                continue
            nxt=b
        elif b==rc:
            if e['oneway_to'] is not None and tuple(e['oneway_to'])!=a:
                continue
            nxt=a
        else:
            continue
        nbrs.append(nxt)
    m=sc['mission']
    lms=collections.defaultdict(list)
    for lm in sc['landmarks']:
        lms[lm['type']].append(tuple(lm['rc']))
    def cands(t,r):
        if t is None:
            return [None]
        if r is not None:
            return [tuple(r['rc'])]
        return lms.get(t,[])
    if m['via'] is not None:
        tl=cands(m['via'],m['via_ref'])
    else:
        tl=cands(m['goal'],m['goal_ref'])
    tl=[t for t in tl if t is not None]
    best=None
    bk=None
    for nxt in nbrs:
        d=min(abs(nxt[0]-t[0])+abs(nxt[1]-t[1]) for t in tl)
        a=rc_to_action(rc,nxt)
        k=(d,REL[(head,a)])
        if bk is None or k<bk:
            bk=k
            best=a
    return best
scenes=json.loads((BASE/'validation/scenes.json').read_text())
labels=json.loads((BASE/'validation/labels.json').read_text())
def cost_for(rid,sc):
    w=sc['weather']
    u=sc['mission']['urgent']
    f=sc['mission']['fragile']
    if rid==0:
        return lambda e:1,False
    if rid==1:
        return lambda e:1+(5 if e['status']=='crowded' else 0),False
    if rid==2:
        return lambda e:0.5 if e['status']=='covered' else 1,False
    if rid==3:
        if w=='rain':
            return lambda e:0.1 if e['status']=='covered' else 1,False
        return lambda e:1+(3 if e['status']=='crowded' else 0),False
    if rid==4:
        return lambda e:1,True
    if rid in (5,8):
        return lambda e:1,False
    if rid==6:
        if u:
            return lambda e:1,False
        return lambda e:1+(5 if e['status']=='crowded' else 0),False
    if rid==7:
        if not f:
            return lambda e:1,False
        return lambda e:1+(5 if e['status']=='crowded' else 0),False
    return None,False
tot=collections.Counter()
hit=collections.Counter()
for si,s in enumerate(scenes):
    for rid in range(10):
        y=labels[si*10+rid]
        if rid==9:
            p=greedy(s)
        else:
            fn,leg=cost_for(rid,s)
            p=pred_cost(s,fn,leg)
        tot[rid]+=1
        if p==y:
            hit[rid]+=1
macro=sum(hit[r]/tot[r] for r in range(10))/10
print(f'VAL oracle TRUE NLP macro={macro:.4f}')
for r in range(10):
    print(f' R{r}:{hit[r]/tot[r]:.3f}',end=' ')
print()
