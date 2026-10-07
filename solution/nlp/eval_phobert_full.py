"""Eval PhoBERT full 6 heads + val end-to-end voi oracle graph."""
import json, pathlib, random, unicodedata, collections, heapq
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import sys
sys.path.insert(0,'/mnt/hdd2/qtech/Phenika-AI/solution/nlp')
from train_phobert_full import MT, DS, coll, LABELS, REFS, BASE
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok=AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
tr=json.loads((BASE/'train/scenes.json').read_text(encoding='utf-8'))
va=json.loads((BASE/'validation/scenes.json').read_text(encoding='utf-8'))
va_labels=json.loads((BASE/'validation/labels.json').read_text())
m=MT().to(dev)
m.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/phobert_mt.pt',map_location=dev))
m.eval()
from torch.utils.data import DataLoader
lva=DataLoader(DS(va,False),batch_size=32,shuffle=False,collate_fn=lambda b:coll(b,tok))
hits={'goal':0,'via':0,'urg':0,'frag':0,'gref':0}; tot=0
pg_all=[]; pv_all=[]; pu_all=[]; pf_all=[]; pr_all=[]
with torch.no_grad():
    for enc,g,v,u,f,r in lva:
        enc={k:v.to(dev) for k,v in enc.items()}
        pg,pv,pu,pf,pr=m(enc)
        pg=pg.argmax(1).cpu(); pv=pv.argmax(1).cpu(); pu=pu.argmax(1).cpu(); pf=pf.argmax(1).cpu(); pr=pr.argmax(1).cpu()
        hits['goal']+= (pg==g).sum().item(); hits['via']+= (pv==v).sum().item(); hits['urg']+= (pu==u).sum().item(); hits['frag']+= (pf==f).sum().item(); hits['gref']+= (pr==r).sum().item()
        tot+=len(g)
        pg_all+=pg.tolist(); pv_all+=pv.tolist(); pu_all+=pu.tolist(); pf_all+=pf.tolist(); pr_all+=pr.tolist()
for k in hits: print(f'PhoBERT val {k}={hits[k]/tot:.4f}')
# end-to-end voi oracle graph + PhoBERT pred (dung true ref khi goal/via dung de upper-bound ref)
REL={('UP',0):0,('UP',3):1,('UP',2):2,('UP',1):3,('DOWN',1):0,('DOWN',2):1,('DOWN',3):2,('DOWN',0):3,('LEFT',2):0,('LEFT',0):1,('LEFT',1):2,('LEFT',3):3,('RIGHT',3):0,('RIGHT',1):1,('RIGHT',0):2,('RIGHT',2):3}
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
INV_L={i:l for i,l in enumerate(LABELS)}
def pred_cost(sc,cf,leg,goal,via):
    adj=build_adj(sc,leg); start=tuple(sc['robot']['rc']); head=sc['robot']['heading']; m=sc['mission']
    lms=collections.defaultdict(list)
    for lm in sc['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    def cands(p,t,tr):
        if p in (None,'NONE',10): return [None]
        if isinstance(p,int): p=INV_L[p] if p<10 else 'NONE'
        if p=='NONE': return [None]
        if p==t and tr is not None: return [tuple(tr['rc'])]
        return lms.get(p,[None])
    vl=cands(via,m['via'],m['via_ref']); gl=cands(goal,m['goal'],m['goal_ref'])
    if via in ('NONE',10): vl=[None]
    best=None;bk=None
    for vr in vl:
        for gr in gl:
            if gr is None: continue
            wps=[w for w in [vr,gr] if w is not None]; cur=start;tot2=0;fst=None;ok=True
            for wp in wps:
                fa=dfirst(adj,cur,wp,cf,head)
                if fa is None and cur!=wp: ok=False;break
                dist={cur:0};pq=[(0,cur)]
                while pq:
                    d,u=heapq.heappop(pq)
                    if d!=dist[u]: continue
                    if u==wp: break
                    for v,e in adj.get(u,[]):
                        nd=d+cf(e)
                        if nd<dist.get(v,1e9): dist[v]=nd;heapq.heappush(pq,(nd,v))
                if wp not in dist: ok=False;break
                tot2+=dist[wp]
                if fst is None: fst=fa
                cur=wp
            if not ok: continue
            k=(tot2,REL[(head,fst)] if fst is not None else 9)
            if bk is None or k<bk: bk=k;best=fst
    return best
def greedy(sc,goal,via):
    rc=tuple(sc['robot']['rc']);head=sc['robot']['heading'];nbrs=[]
    for e in sc['edges']:
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
    m=sc['mission'];lms=collections.defaultdict(list)
    for lm in sc['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    def cands(p,t,tr):
        if p in (None,'NONE',10): return [None]
        if isinstance(p,int): p=INV_L[p] if p<10 else 'NONE'
        if p=='NONE': return [None]
        if p==t and tr is not None: return [tuple(tr['rc'])]
        return lms.get(p,[None])
    if via not in ('NONE',10): tl=cands(via,m['via'],m['via_ref'])
    else: tl=cands(goal,m['goal'],m['goal_ref'])
    tl=[t for t in tl if t is not None]
    best=None;bk=None
    for nxt in nbrs:
        d=min(abs(nxt[0]-t[0])+abs(nxt[1]-t[1]) for t in tl)
        a=rc_to_action(rc,nxt);k=(d,REL[(head,a)])
        if bk is None or k<bk: bk=k;best=a
    return best
def cost_for(rid,u,f,w):
    if rid==0: return lambda e:1,False
    if rid==1: return lambda e:1+(5 if e['status']=='crowded' else 0),False
    if rid==2: return lambda e:0.5 if e['status']=='covered' else 1,False
    if rid==3: return (lambda e:0.1 if e['status']=='covered' else 1) if w=='rain' else (lambda e:1+(3 if e['status']=='crowded' else 0)),False
    if rid==4: return lambda e:1,True
    if rid in (5,8): return lambda e:1,False
    if rid==6: return (lambda e:1) if u else (lambda e:1+(5 if e['status']=='crowded' else 0)),False
    if rid==7: return (lambda e:1) if not f else (lambda e:1+(5 if e['status']=='crowded' else 0)),False
    return None,False
per_tot=collections.Counter();per_hit=collections.Counter()
for si,s in enumerate(va):
    g,pv,pu,pf=pg_all[si],pv_all[si],pu_all[si],pf_all[si]
    for rid in range(10):
        y=va_labels[si*10+rid]
        if rid==9: p=greedy(s,g,pv)
        else:
            fn,leg=cost_for(rid,pu,pf,s['weather']); p=pred_cost(s,fn,leg,g,pv)
        per_tot[rid]+=1
        if p==y: per_hit[rid]+=1
macro=sum(per_hit[r]/per_tot[r] for r in range(10))/10
print(f'VAL end-to-end (oracle graph + PhoBERT): macro={macro:.4f}')
for r in range(10): print(f' R{r}:{per_hit[r]/per_tot[r]:.3f}',end=' ')
print()
