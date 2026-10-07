"""Review 3 vong + val end-to-end voi NLP du doan (graph oracle)."""
import json, pathlib, collections, heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
REL={('UP',0):0,('UP',3):1,('UP',2):2,('UP',1):3,
     ('DOWN',1):0,('DOWN',2):1,('DOWN',3):2,('DOWN',0):3,
     ('LEFT',2):0,('LEFT',0):1,('LEFT',1):2,('LEFT',3):3,
     ('RIGHT',3):0,('RIGHT',1):1,('RIGHT',0):2,('RIGHT',2):3}
def rc_to_action(a,b):
    return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((b[0]-a[0],b[1]-a[1]))
def build_adj(scene, legged=False):
    import collections
    adj=collections.defaultdict(list)
    for e in scene['edges']:
        a,b=tuple(e['a']),tuple(e['b'])
        if e['status']=='closed': continue
        if e['stairs'] and not legged: continue
        if e['oneway_to'] is not None:
            t=tuple(e['oneway_to'])
            if t==b: adj[a].append((b,e))
            elif t==a: adj[b].append((a,e))
        else: adj[a].append((b,e)); adj[b].append((a,e))
    return adj
def dfirst(adj,s,t,cf,hd):
    pq=[]
    for v,e in adj.get(s,[]):
        a=rc_to_action(s,v); heapq.heappush(pq,(cf(e),REL[(hd,a)],v,a))
    vis={}
    while pq:
        c,r,u,fa=heapq.heappop(pq)
        if u in vis: continue
        vis[u]=1
        if u==t: return fa
        for v,e in adj.get(u,[]):
            if v in vis: continue
            heapq.heappush(pq,(c+cf(e),r,v,fa))
    return None
def pred_cost(scene,cf,legged=False,goal=None,via=None,urg=None,frg=None):
    # goal/via la string du doan (hoac None->NONE); dung ref that de don gian (vong sau se du doan ref)
    adj=build_adj(scene,legged)
    start=tuple(scene['robot']['rc']); head=scene['robot']['heading']
    lms=collections.defaultdict(list)
    for lm in scene['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    # resolve with PREDICTED goal/via but TRUE ref rc (upper-bound for ref)
    m=scene['mission']
    def cands(ptyp, true_typ, true_ref):
        if ptyp in (None,'NONE'): return [None]
        if true_typ==ptyp and true_ref is not None: return [tuple(true_ref['rc'])]
        lst=lms.get(ptyp,[])
        return lst if lst else [None]
    via_list=cands(via, m['via'], m['via_ref']); goal_list=cands(goal, m['goal'], m['goal_ref'])
    if via in (None,'NONE'): via_list=[None]
    best=None; bk=None
    for vr in via_list:
        for gr in goal_list:
            if gr is None: continue
            wps=[w for w in [vr,gr] if w is not None]
            cur=start; tot=0; first=None; ok=True
            for wp in wps:
                fa=dfirst(adj,cur,wp,cf,head)
                if fa is None and cur!=wp: ok=False; break
                dist={cur:0}; pq=[(0,cur)]
                while pq:
                    d,u=heapq.heappop(pq)
                    if d!=dist[u]: continue
                    if u==wp: break
                    for v,e in adj.get(u,[]):
                        nd=d+cf(e)
                        if nd<dist.get(v,1e9): dist[v]=nd; heapq.heappush(pq,(nd,v))
                if wp not in dist: ok=False; break
                tot+=dist[wp]
                if first is None: first=fa
                cur=wp
            if not ok: continue
            k=(tot, REL[(head,first)] if first is not None else 9)
            if bk is None or k<bk: bk=k; best=first
    return best

# train TF-IDF cho goal/via/urg/frag
tr_s=json.loads((BASE/'train/scenes.json').read_text(encoding='utf-8'))
va_s=json.loads((BASE/'validation/scenes.json').read_text(encoding='utf-8'))
va_labels=json.loads((BASE/'validation/labels.json').read_text())
Xtr=[s['mission']['text'] for s in tr_s]; Xva=[s['mission']['text'] for s in va_s]
vec=TfidfVectorizer(max_features=8000, ngram_range=(1,2))
Xtrv=vec.fit_transform(Xtr); Xvav=vec.transform(Xva)
pred_goal, pred_via, pred_urg, pred_frag = {},{},{},{}
for key, ytr in [('goal',[s['mission']['goal'] for s in tr_s]),('via',[s['mission']['via'] or 'NONE' for s in tr_s]),('urg',[int(s['mission']['urgent']) for s in tr_s]),('frag',[int(s['mission']['fragile']) for s in tr_s])]:
    clf=LogisticRegression(max_iter=1000,C=4); clf.fit(Xtrv,ytr)
    pv=clf.predict(Xvav)
    if key=='goal': pred_goal=pv
    elif key=='via': pred_via=pv
    elif key=='urg': pred_urg=pv
    else: pred_frag=pv

def cost_for(rid, urg, frag, wthr):
    if rid==0: return lambda e:1, False
    if rid==1: return lambda e:1+(5 if e['status']=='crowded' else 0), False
    if rid==2: return lambda e:0.5 if e['status']=='covered' else 1, False
    if rid==3: return (lambda e:0.1 if e['status']=='covered' else 1) if wthr=='rain' else (lambda e:1+(3 if e['status']=='crowded' else 0)), False
    if rid==4: return lambda e:1, True
    if rid==5: return lambda e:1, False
    if rid==6: return (lambda e:1) if urg else (lambda e:1+(5 if e['status']=='crowded' else 0)), False
    if rid==7: return (lambda e:1) if not frag else (lambda e:1+(5 if e['status']=='crowded' else 0)), False
    if rid==8: return lambda e:1, False
    return None, False

# greedy R9 manhattan voi NLP pred
import math
def greedy_pred(scene, goal, via):
    rc=tuple(scene['robot']['rc']); head=scene['robot']['heading']
    nbrs=[]
    for e in scene['edges']:
        a,b=tuple(e['a']),tuple(e['b'])
        if e['status']=='closed' or e['stairs']: continue
        nxt=None
        if a==rc:
            if e['oneway_to'] is not None and tuple(e['oneway_to'])!=b: continue
            nxt=b
        elif b==rc:
            if e['oneway_to'] is not None and tuple(e['oneway_to'])!=a: continue
            nxt=a
        else: continue
        nbrs.append(nxt)
    lms=collections.defaultdict(list)
    for lm in scene['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    m=scene['mission']
    # dung pred goal/via + true ref khi trung
    def cands(p, t, tr):
        if p in (None,'NONE'): return [None]
        if p==t and tr is not None: return [tuple(tr['rc'])]
        return lms.get(p,[None])
    tl=cands(via if via!='NONE' else None, m['via'], m['via_ref']) if via!='NONE' else cands(goal, m['goal'], m['goal_ref'])
    # neu via pred NONE -> target goal
    if via=='NONE': tl=cands(goal, m['goal'], m['goal_ref'])
    tl=[t for t in tl if t is not None]
    if not tl or not nbrs: return None
    best=None; bk=None
    for nxt in nbrs:
        d=min(abs(nxt[0]-t[0])+abs(nxt[1]-t[1]) for t in tl)
        a=rc_to_action(rc,nxt); k=(d,REL[(head,a)])
        if bk is None or k<bk: bk=k; best=a
    return best

per_r_tot=collections.Counter(); per_r_hit=collections.Counter()
for si,s in enumerate(va_s):
    g,pv,pu,pf = pred_goal[si], pred_via[si], int(pred_urg[si]), int(pred_frag[si])
    for rid in range(10):
        y=va_labels[si*10+rid]
        if rid==9: p=greedy_pred(s,g,pv)
        else:
            fn,leg=cost_for(rid,pu,pf,s['weather']); p=pred_cost(s,fn,leg,g,pv,pu,pf)
        per_r_tot[rid]+=1
        if p==y: per_r_hit[rid]+=1
macro=sum(per_r_hit[r]/per_r_tot[r] for r in range(10))/10
print(f'VAL end-to-end (oracle graph + TFIDF NLP): macro={macro:.4f}')
for r in range(10):
    print(f' R{r}: {per_r_hit[r]/per_r_tot[r]:.3f}', end=' ')
print()
