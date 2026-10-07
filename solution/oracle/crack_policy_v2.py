"""Crack R4/R5/R7/R8: grid-search turn-cost + fragile/stairs variants. Chi dung train (compliant)."""
import json, pathlib, collections, heapq, itertools
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
REL={('UP',0):0,('UP',3):1,('UP',2):2,('UP',1):3,('DOWN',1):0,('DOWN',2):1,('DOWN',3):2,('DOWN',0):3,('LEFT',2):0,('LEFT',0):1,('LEFT',1):2,('LEFT',3):3,('RIGHT',3):0,('RIGHT',1):1,('RIGHT',0):2,('RIGHT',2):3}
HEAD2DIR={'UP':0,'DOWN':1,'LEFT':2,'RIGHT':3}
DIR2HEAD={v:k for k,v in HEAD2DIR.items()}
def rc_to_action(a,b): return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((b[0]-a[0],b[1]-a[1]))
def build_adj(sc,leg):
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
def ttype(prev_dir, new_action):
    if prev_dir==new_action: return 'S'
    if (prev_dir,new_action) in [(0,3),(3,1),(1,2),(2,0)]: return 'R'
    if (prev_dir,new_action) in [(0,2),(2,1),(1,3),(3,0)]: return 'L'
    return 'B'
def dfull_turn(adj,start,head_dir,target,edge_cost,turn_cost):
    # state: (node, incoming_dir). Start: virtual heading=head_dir
    import heapq
    pq=[]
    for v,e in adj.get(start,[]):
        a=rc_to_action(start,v)
        c=edge_cost(e)+turn_cost[ttype(head_dir,a)]
        heapq.heappush(pq,(c,REL[(DIR2HEAD[head_dir],a)],v,a,a))
    vis={}
    while pq:
        c,r,u,din,fa=heapq.heappop(pq)
        if (u,din) in vis: continue
        vis[(u,din)]=1
        if u==target: return fa
        for v,e in adj.get(u,[]):
            a2=rc_to_action(u,v)
            if (v,a2) in vis: continue
            nc=c+edge_cost(e)+turn_cost[ttype(din,a2)]
            heapq.heappush(pq,(nc,r,v,a2,fa))
    return None
def predict_turn(sc,edge_cost,leg,turn_cost):
    adj=build_adj(sc,leg); start=tuple(sc['robot']['rc']); hd=HEAD2DIR[sc['robot']['heading']]
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
                fa=dfull_turn(adj,cur,cur_dir,wp,edge_cost,turn_cost)
                if fa is None and cur!=wp: ok=False; break
                # distance with turn for waypoint selection (approx: run dijkstra dist with turn)
                # simplified: BFS dist with edge+turn via state search
                import heapq as hq
                dist={(cur,cur_dir):0}; pq=[(0,cur,cur_dir)]
                found=None
                while pq:
                    d,u,dd=hq.heappop(pq)
                    if d!=dist[(u,dd)]: continue
                    if u==wp: found=d; break
                    for v,e in adj.get(u,[]):
                        a2=rc_to_action(u,v); nd=d+edge_cost(e)+turn_cost[ttype(dd,a2)]
                        if nd<dist.get((v,a2),1e9): dist[(v,a2)]=nd; hq.heappush(pq,(nd,v,a2))
                if found is None and cur!=wp: ok=False; break
                total+=found if found else 0
                if first is None: first=fa
                # update cur_dir: need last dir? approx keep fa? For multi-waypoint, recompute heading as direction into wp (skip precise)
                cur=wp
                # crude: keep cur_dir unchanged (via rare) – acceptable for grid-search ranking
            if not ok: continue
            k=(total,REL[(sc['robot']['heading'],first)] if first is not None else 9)
            if bk is None or k<bk: bk=k; best=first
    return best
scenes=json.loads((BASE/'train/scenes.json').read_text())
labels=json.loads((BASE/'train/labels.json').read_text())
def acc(rid, fn):
    hit=tot=0
    for i,s in enumerate(scenes):
        p=fn(s)
        if p is None: continue
        tot+=1
        if p==labels[i*10+rid]: hit+=1
    return hit/max(1,tot)
base=lambda e:1
print("=== R5/R8 turn grid-search (train full) ===",flush=True)
for name,rid in [('R5',5),('R8',8)]:
    best=[]
    for R in [0,0.5,1,2]:
        for L in [1,2,3,4]:
            for B in [4,6,8,10]:
                tc={'S':0,'R':R,'L':L,'B':B}
                # R8 favors Right: swap R/L search too
                a=acc(rid, lambda s,tc=tc: predict_turn(s,base,False,tc))
                best.append((a,R,L,B))
    best.sort(reverse=True)
    for a,R,L,B in best[:5]:
        print(f' {name} S=0 R={R} L={L} B={B} -> {a:.4f}',flush=True)
print("=== R7 fragile variants ===",flush=True)
for fname,fn in [
    ('frag_avoid_crowded5', lambda e:1+(5 if e['status']=='crowded' else 0)),
    ('frag_avoid_crowded3', lambda e:1+(3 if e['status']=='crowded' else 0)),
    ('frag_avoid_crowded2', lambda e:1+(2 if e['status']=='crowded' else 0)),
    ('frag_turn_R1L2B6', None),]:
    if fn is not None:
        # fragile=True uses fn, else base; need mission fragile
        def pred(s,fn=fn):
            f=s['mission']['fragile']
            ec=fn if f else base
            return predict_turn(s,ec,False,{'S':0,'R':0,'L':0,'B':0})
        print(f' R7-{fname}: {acc(7,pred):.4f}',flush=True)
# R7 turn hypothesis: fragile = it re
for tc in [{'S':0,'R':1,'L':2,'B':6},{'S':0,'R':1,'L':1,'B':8},{'S':0,'R':2,'L':2,'B':8}]:
    def pred2(s,tc=tc):
        if s['mission']['fragile']: return predict_turn(s,base,False,tc)
        return predict_turn(s,base,False,{'S':0,'R':0,'L':0,'B':0})
    print(f' R7-turn {tc}: {acc(7,pred2):.4f}',flush=True)
print("=== R4 variants ===",flush=True)
for desc,leg,ec in [('leg_base',True,base),('leg_avoid_crowded',True,lambda e:1+(3 if e['status']=='crowded' else 0)),('leg_likeR2',True,lambda e:0.5 if e['status']=='covered' else 1)]:
    print(f' R4-{desc}: {acc(4, lambda s,leg=leg,ec=ec: predict_turn(s,ec,leg,{"S":0,"R":0,"L":0,"B":0})):.4f}',flush=True)
print("DONE")
