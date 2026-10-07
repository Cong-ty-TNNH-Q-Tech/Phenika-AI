"""Vong 1: Simulator oracle + R0 BFS baseline."""
import json, pathlib, collections, heapq

BASE = pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
ACT = {'UP':0,'DOWN':1,'LEFT':2,'RIGHT':3}
def rc_to_action(a,b):
    dr, dc = b[0]-a[0], b[1]-a[1]
    if dr==-1 and dc==0: return 0
    if dr==1 and dc==0: return 1
    if dr==0 and dc==-1: return 2
    if dr==0 and dc==1: return 3
    return None

def build_adj(scene, legged=False):
    adj = collections.defaultdict(list)  # rc-tuple -> list of (nbr, edge)
    for e in scene['edges']:
        a, b = tuple(e['a']), tuple(e['b'])
        if e['status']=='closed': continue
        if e['stairs'] and not legged: continue
        # directed handling
        if e['oneway_to'] is not None:
            t = tuple(e['oneway_to'])
            if t==b: adj[a].append((b,e))
            elif t==a: adj[b].append((a,e))
            else: pass
        else:
            adj[a].append((b,e)); adj[b].append((a,e))
    return adj

def shortest_first_step(scene, legged=False, cost_fn=None):
    """Dijkstra from robot to target (via then goal). Return first action or None."""
    adj = build_adj(scene, legged)
    start = tuple(scene['robot']['rc'])
    m = scene['mission']
    # resolve targets
    lms = collections.defaultdict(list)
    for lm in scene['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    def resolve(typ, ref):
        if typ is None: return None
        if ref is not None: return tuple(ref['rc'])
        cands = lms.get(typ, [])
        if len(cands)==1: return cands[0]
        return None  # ambiguous -> skip for clean eval
    via_rc = resolve(m['via'], m['via_ref'])
    goal_rc = resolve(m['goal'], m['goal_ref'])
    if goal_rc is None: return None, 'ambig_goal'
    waypoints = [w for w in [via_rc, goal_rc] if w is not None]
    # multi-leg Dijkstra
    cur = start
    first_step = None
    for wp in waypoints:
        # dijkstra
        dist={cur:0}; prev={}; pq=[(0,cur)]
        seen=set()
        while pq:
            d,u = heapq.heappop(pq)
            if u in seen: continue
            seen.add(u)
            if u==wp: break
            for v,e in adj.get(u,[]):
                c = cost_fn(e) if cost_fn else 1
                nd = d+c
                if nd < dist.get(v,1e9):
                    dist[v]=nd; prev[v]=u; heapq.heappush(pq,(nd,v))
        if wp not in dist: return None, 'unreachable'
        # reconstruct first step of this leg (only first leg matters)
        if first_step is None:
            path=[wp]
            while path[-1]!=cur:
                path.append(prev[path[-1]])
            path=path[::-1]
            if len(path)<2: return None, 'stay'
            first_step = rc_to_action(cur, path[1])
        cur = wp
    return first_step, 'ok'

for split in ['train','validation']:
    scenes=json.loads((BASE/split/'scenes.json').read_text())
    labels=json.loads((BASE/split/'labels.json').read_text())
    obs=json.loads((BASE/split/'observations.json').read_text())
    # R0 only, clean scenes (single goal, no via ambiguity, reachable)
    tot=clean=hit=0; reach_fail=0; ambig=0
    per_r_hit = collections.Counter(); per_r_tot=collections.Counter()
    for i,s in enumerate(scenes):
        for r in range(10):
            y = labels[i*10+r]
            per_r_tot[r]+=1
        # compute R0 BFS pred
        pred,_ = shortest_first_step(s, legged=False)
        y0 = labels[i*10+0]
        tot+=1
        if pred is None:
            if _ in ('ambig_goal',): ambig+=1
            else: reach_fail+=1
            continue
        clean+=1
        if pred==y0: hit+=1
    print(f'{split}: scenes={len(scenes)} clean_reachable={clean}/{tot} ambig_or_unreach={ambig+reach_fail} R0-BFS-acc-on-clean={hit/max(1,clean):.3f} overall={hit/tot:.3f}')
    # detail: how many clean scenes have duplicate goal without ref?
    dup_noref = sum(1 for s in scenes if sum(1 for lm in s['landmarks'] if lm['type']==s['mission']['goal'])>1 and s['mission']['goal_ref'] is None)
    print(f'  dup-goal-no-ref scenes: {dup_noref}')
