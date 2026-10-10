"""Policy v5: best-of train/val. B-first tie-breaks, crowd/cover/turn costs, R7 fragile split, R9 euc-greedy."""
import collections, heapq, math

def rc_to_action(a, b):
    return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((b[0]-a[0], b[1]-a[1]))

def rel(hd, a):
    if a == hd: return 'S'
    if (hd, a) in [(0,3),(3,1),(1,2),(2,0)]: return 'R'
    if (hd, a) in [(0,2),(2,1),(1,3),(3,0)]: return 'L'
    return 'B'

HEAD = {'UP':0,'DOWN':1,'LEFT':2,'RIGHT':3}
ZERO = {'S':0,'R':0,'L':0,'B':0}
T1 = {'S':0,'R':1,'L':1,'B':3}
T2 = {'S':0,'R':1,'L':1,'B':5}
T_R5 = {'S':0,'R':2,'L':2,'B':20}

def build_adj(edges, leg=False):
    adj = collections.defaultdict(list)
    for e in edges:
        a, b = tuple(e['a']), tuple(e['b'])
        if e['status'] == 'closed': continue
        if e['stairs'] and not leg: continue
        t = e['oneway_to']
        if t is not None:
            t = tuple(t)
            if t == b: adj[a].append((b, e))
            elif t == a: adj[b].append((a, e))
        else:
            adj[a].append((b, e)); adj[b].append((a, e))
    return adj

def _solve(adj, cur, target, cfn, hdc, tbr, tc):
    pq = []
    for v, e in adj.get(cur, []):
        x = rc_to_action(cur, v)
        heapq.heappush(pq, (cfn(e) + tc[rel(hdc, x)], tbr[rel(hdc, x)], v, x, x))
    vis = {}
    while pq:
        c, r, u, di, fa = heapq.heappop(pq)
        if (u, di) in vis: continue
        vis[(u, di)] = 1
        if u == target: return fa, c
        for v, e in adj.get(u, []):
            a2 = rc_to_action(u, v)
            if (v, a2) in vis: continue
            heapq.heappush(pq, (c + cfn(e) + tc[rel(di, a2)], r, v, a2, fa))
    return None, None

def _targets(m, lms):
    if m['via'] is not None:
        vl = [tuple(m['via_ref']['rc'])] if m['via_ref'] else lms.get(m['via'], [])
    else:
        vl = [None]
    gl = [tuple(m['goal_ref']['rc'])] if m['goal_ref'] else lms.get(m['goal'], [])
    return vl, gl

def predict_cost(sc, cfn, leg, tb, tc):
    start = tuple(sc['robot']['rc']); hd = HEAD[sc['robot']['heading']]
    m = sc['mission']
    lms = collections.defaultdict(list)
    for lm in sc['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    vl, gl = _targets(m, lms)
    adj = build_adj(sc['edges'], leg)
    tbr = {d: i for i, d in enumerate(tb)}
    best = None; bk = None
    for vr in vl:
        for gr in gl:
            if gr is None: continue
            wps = [w for w in [vr, gr] if w is not None]
            cur = start; tot = 0; first = None; ok = True
            for wp in wps:
                if cur == wp: continue
                fa, c = _solve(adj, cur, wp, cfn, hd, tbr, tc)
                if fa is None: ok = False; break
                tot += c
                if first is None: first = fa
                cur = wp
            if not ok: continue
            k = (tot, tbr[rel(hd, first)] if first is not None else 99)
            if bk is None or k < bk: bk = k; best = first
    return best

def greedy9(sc, tb='BSLR', cp=2.0):
    adj = build_adj(sc['edges'], False)
    start = tuple(sc['robot']['rc']); hd = HEAD[sc['robot']['heading']]
    m = sc['mission']
    lms = collections.defaultdict(list)
    for lm in sc['landmarks']: lms[lm['type']].append(tuple(lm['rc']))
    tl = ([tuple(m['via_ref']['rc'])] if m['via_ref'] else lms.get(m['via'], [])) if m['via'] \
         else ([tuple(m['goal_ref']['rc'])] if m['goal_ref'] else lms.get(m['goal'], []))
    tl = [x for x in tl if x is not None]
    if not tl: return None
    tbr = {d: i for i, d in enumerate(tb)}
    best = None; bk = None
    for v, e in adj.get(start, []):
        a = rc_to_action(start, v)
        d = min(math.hypot(v[0]-t[0], v[1]-t[1]) for t in tl)
        if e['status'] == 'crowded': d += cp
        k = (d, tbr[rel(hd, a)])
        if bk is None or k < bk: bk = k; best = a
    return best

# (cfn, leg, tb, tc) per robot; R1 style-split, R3 goal-split, R7 fragile split
GLL_GOALS = {'lab','lecture','library'}
def params_for(rid, mission, weather, style=None, goal=None):
    urg = mission['urgent']; frag = mission['fragile']
    if rid == 0:
        return (lambda e: 1), False, 'BLRS', ZERO
    if rid == 1:
        if style == 'night':
            return (lambda e: 1), False, 'SRLB', ZERO
        return (lambda e: 1 + (2 if e['status']=='crowded' else 0)), False, 'SRLB', ZERO
    if rid == 2:
        return (lambda e: 1 + (1 if e['status']=='crowded' else 0)), False, 'BLRS', ZERO
    if rid == 3:
        if goal in GLL_GOALS:
            return (lambda e: 1), False, 'SBRL', {'S':0,'R':2,'L':2,'B':10}
        c1c5 = lambda e: (0.5 if e['status']=='covered' else 1) + (1 if e['status']=='crowded' else 0)
        return c1c5, False, 'SBRL', ZERO
    if rid == 4:
        return (lambda e: 1 + (1 if e['status']=='crowded' else 0)), True, 'SRLB', ZERO
    if rid == 5:
        if style == 'night':
            return (lambda e: 1), False, 'SBRL', ZERO
        c1c5 = lambda e: (0.5 if e['status']=='covered' else 1) + (1 if e['status']=='crowded' else 0)
        return c1c5, False, 'SRLB', T_R5
    if rid == 6:
        return (lambda e: 1 + (1 if e['status']=='crowded' else 0)), False, 'LRSB', T2
    if rid == 7:
        if frag:
            return (lambda e: 1 + (3 if e['status']=='crowded' else 0)), False, 'SBRL', {'S':0,'R':1,'L':1,'B':10}
        return (lambda e: 1 + (1 if e['status']=='crowded' else 0)), False, 'SBRL', ZERO
    if rid == 8:
        return (lambda e: 1 + (2 if e['status']=='crowded' else 0)), False, 'RSLB', ZERO
    raise ValueError(rid)

def predict_all(sc):
    out = {}
    for rid in range(9):
        cfn, leg, tb, tc = params_for(rid, sc['mission'], sc['weather'], sc.get('style'), sc['mission']['goal'])
        out[rid] = predict_cost(sc, cfn, leg, tb, tc)
    out[9] = greedy9(sc)
    return out
