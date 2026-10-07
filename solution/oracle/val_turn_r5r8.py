"""Val best: PhoBERT + turn R5/R8 cho no-via, base cho via."""
import json, pathlib, collections, heapq
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
# load PhoBERT preds tu eval truoc? chay lai nhanh: dung TF-IDF? Khong, dung PhoBERT 92% goal.
# De tiet kiem time: dung TRUE goal/via + PhoBERT urg/frag? Actually goal la quan trong nhat (92% vs 100%). Dung true goal de upper-bound turn?
# Tot nhat: dung PhoBERT goal 92% + turn. Nhung de nhanh, dung true mission (oracle NLP) + turn de do ceiling turn.
REL={('UP',0):0,('UP',3):1,('UP',2):2,('UP',1):3,('DOWN',1):0,('DOWN',2):1,('DOWN',3):2,('DOWN',0):3,('LEFT',2):0,('LEFT',0):1,('LEFT',1):2,('LEFT',3):3,('RIGHT',3):0,('RIGHT',1):1,('RIGHT',0):2,('RIGHT',2):3}
HEAD2DIR={'UP':0,'DOWN':1,'LEFT':2,'RIGHT':3}
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
def ttype(p,n):
    if p==n: return 'S'
    if (p,n) in [(0,3),(3,1),(1,2),(2,0)]: return 'R'
    if (p,n) in [(0,2),(2,1),(1,3),(3,0)]: return 'L'
    return 'B'
def dfull(adj,s,hd,t,ef,tc):
    pq=[]
    for v,e in adj.get(s,[]):
        a=rc_to_action(s,v)
        heapq.heappush(pq,(ef(e)+tc[ttype(hd,a)],REL[({0:'UP',1:'DOWN',2:'LEFT',3:'RIGHT'}[hd],a)],v,a,a))
    vis={}
    while pq:
        c,r,u,din,fa=heapq.heappop(pq)
        if (u,din) in vis: continue
        vis[(u,din)]=1
        if u==t: return fa
        for v,e in adj.get(u,[]):
            a2=rc_to_action(u,v)
            if (v,a2) in vis: continue
            heapq.heappush(pq,(c+ef(e)+tc[ttype(din,a2)],r,v,a2,fa))
    return None
scenes=json.loads((BASE/'validation/scenes.json').read_text())
labels=json.loads((BASE/'validation/labels.json').read_text())
R5TC={'S':0,'R':1,'L':1,'B':8}; R8TC={'S':0,'R':0,'L':3,'B':6}
base=lambda e:1
for rid,tc,name in [(5,R5TC,'R5-turn'),(8,R8TC,'R8-turn')]:
    tot=hit=0; tot_novia=hit_novia=0
    for i,s in enumerate(scenes):
        adj=build_adj(s,False); st=tuple(s['robot']['rc']); hd=HEAD2DIR[s['robot']['heading']]; m=s['mission']
        lms=collections.defaultdict(list)
        for lm in s['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
        # chi no-via single-goal de test turn sach
        if m['via'] is not None: continue
        if sum(1 for lm in s['landmarks'] if lm['type']==m['goal'])!=1: continue
        # target true
        tgt=lms[m['goal']][0]
        p=dfull(adj,st,hd,tgt,base,tc)
        tot+=1
        if p==labels[i*10+rid]: hit+=1
    print(f'{name} val clean no-via single: {hit}/{tot}={hit/max(1,tot):.4f}')
