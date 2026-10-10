"""Learn a legal-action ranker from train scene annotations.

Shortest path features track heading through via using a two-stage graph.
Test images/text are never used to choose features or fit a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    from .oracle_xgb import adjacency, locations, relative, HEAD, ACTIONS
except ImportError:
    from oracle_xgb import adjacency, locations, relative, HEAD, ACTIONS

TYPES = ['library', 'dorm', 'sports', 'clinic', 'canteen', 'parking',
         'lecture', 'lab', 'office', 'gate']
STYLES = ['classic', 'night', 'print', 'sketch']
STATUS = ['normal', 'crowded', 'covered']
REFS = ['NONE', 'north', 'south', 'west', 'east', 'near', 'far',
        'north_most', 'south_most', 'west_most', 'east_most', 'anchor_near']
# A compact hypothesis bank rather than fitting unrestricted policies on 300 val scenes.
BANK = [(crowd, cover, turn, back, right) for crowd, cover in
        [(1, 1), (2, 1), (3, 1), (6, 1), (11, 1), (1, .5), (1, .15),
         (2, .5), (3, .5), (6, .5), (2, 2), (3, 2)]
        for turn, back, right in [(0, 0, 1), (1, 5, 1), (2, 20, 1)]]
BANK += [(2, .5, .5, 10, .1), (3, 1, 3, 20, .1), (1, 1, 2, 10, .25),
         (1, 1, 1, 5, 3), (2, 1, 1, 5, 1), (1, 1, 5, 50, 1)]
VERSION = 'heading-via-ranker-3'
CACHE_VERSION = 'heading-via-ranker-2'


def route_scores(scene, legged, config, first_leg_only=False):
    """Cost for each forced first action, INF if no legal route exists."""
    adj = adjacency(scene, legged)
    nodes = sorted({tuple(n['rc']) for n in scene['nodes']} |
                   {v for u in adj for v, _ in adj[u]} | set(adj))
    index = {n: i for i, n in enumerate(nodes)}
    start = tuple(scene['robot']['rc'])
    if start not in index:
        return np.full(4, np.inf)
    heading = HEAD[scene['robot']['heading']]
    via, goals = locations(scene, 'via'), locations(scene, 'goal')
    via = {v for v in via if v is not None and v in index}
    goals = {g for g in goals if g is not None and g in index}
    if first_leg_only and via:
        goals, via = via, set()
    has_via = bool(via)
    stride = len(nodes) * 4
    # A sink allows all goal headings and duplicate goal instances in one solve.
    sink = stride * (2 if has_via else 1)
    rr, cc, values = [], [], []
    crowd, cover, turn, back, right = config[:5]
    stair_penalty = config[5] if len(config) > 5 else 0.
    weights = {'normal': 1., 'crowded': crowd, 'covered': cover}
    for phase in range(2 if has_via else 1):
        offset = phase * stride
        for node in nodes:
            for hd in range(4):
                source = offset + index[node] * 4 + hd
                # Once a waypoint is visited, transition without resetting heading.
                if has_via and phase == 0 and node in via:
                    rr.append(source); cc.append(stride + index[node] * 4 + hd); values.append(0.)
                    continue
                if (not has_via or phase == 1) and node in goals:
                    rr.append(source); cc.append(sink); values.append(0.)
                for nxt, edge in adj[node]:
                    action = ACTIONS[(nxt[0] - node[0], nxt[1] - node[1])]
                    rel = relative(hd, action)
                    penalty = [0., turn * right, turn, back][rel]
                    rr.append(source); cc.append(offset + index[nxt] * 4 + action)
                    values.append(weights.get(edge['status'], 1.) + penalty + stair_penalty * bool(edge['stairs']))
    graph = csr_matrix((values, (rr, cc)), shape=(sink + 1, sink + 1))
    distances = dijkstra(graph.T.tocsr(), directed=True, indices=sink)
    scores = np.full(4, np.inf)
    phase = 1 if has_via and start in via else 0
    for nxt, edge in adj[start]:
        action = ACTIONS[(nxt[0] - start[0], nxt[1] - start[1])]
        rel = relative(heading, action)
        cost = weights.get(edge['status'], 1.) + [0., turn * right, turn, back][rel] + stair_penalty * bool(edge['stairs'])
        scores[action] = cost + distances[phase * stride + index[nxt] * 4 + action]
    return scores


def normalized_costs(scores):
    finite = np.isfinite(scores)
    minimum = np.min(scores[finite]) if finite.any() else 0.
    difference = np.where(finite, scores - minimum, 99.)
    return np.column_stack([np.minimum(np.nan_to_num(scores, posinf=99.), 99.),
                            np.minimum(difference, 99.),
                            difference / max(1., minimum), (difference < 1e-7)])


def scene_features(scene, legged=False):
    m = scene['mission']
    heading = HEAD[scene['robot']['heading']]
    start = tuple(scene['robot']['rc'])
    adj = adjacency(scene, legged)
    edge_by_action = {ACTIONS[(v[0] - start[0], v[1] - start[1])]: (v, e)
                      for v, e in adj[start]}
    global_features = [m['urgent'], m['fragile'], scene['weather'] == 'rain',
                       m['via'] is not None, len(scene['landmarks'])]
    global_features += [scene.get('style') == x for x in STYLES]
    global_features += [m['goal'] == x for x in TYPES]
    global_features += [m['via'] == x for x in TYPES]
    for key in ['goal_ref', 'via_ref']:
        ref = m.get(key) or {}
        global_features += [ref.get('kind', 'NONE') == x for x in REFS]
    via, goal = locations(scene, 'via'), locations(scene, 'goal')
    targets = [v for v in via if v is not None] or [g for g in goal if g is not None]
    local = []
    for action in range(4):
        pair = edge_by_action.get(action)
        nxt, edge = pair if pair else (start, {})
        rel = relative(heading, action)
        feat = global_features + [pair is not None] + [rel == x for x in range(4)]
        feat += [edge.get('status') == x for x in STATUS]
        feat += [edge.get('stairs', False), edge.get('oneway_to') is not None]
        feat += [len(adj[nxt]), len(adj[start])]
        if targets:
            man = [abs(nxt[0] - t[0]) + abs(nxt[1] - t[1]) for t in targets]
            eu = [math.hypot(nxt[0] - t[0], nxt[1] - t[1]) for t in targets]
            feat += [min(man), min(eu), len(targets)]
        else:
            feat += [99, 99, 0]
        local.append(feat)
    features = [np.array(local, dtype=np.float32)]
    for config in BANK:
        features.append(normalized_costs(route_scores(scene, legged, config)))
    # Compare joint optimization with minimizing the first leg only.
    for config in BANK[::3]:
        features.append(normalized_costs(route_scores(scene, legged, config, True)))
    return np.concatenate(features, axis=1).astype(np.float32)


def cached_features(scenes, dest):
    digest = hashlib.sha256((CACHE_VERSION + json.dumps(scenes, sort_keys=True)).encode()).hexdigest()
    if dest.exists():
        with np.load(dest) as loaded:
            if str(loaded['digest']) == digest:
                return loaded['feet'], loaded['legs']
    feet, legs = [], []
    for i, scene in enumerate(scenes):
        feet.append(scene_features(scene)); legs.append(scene_features(scene, True))
        if (i + 1) % 100 == 0:
            print(f'features {i + 1}/{len(scenes)}', flush=True)
    feet, legs = np.array(feet), np.array(legs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, feet=feet, legs=legs, digest=digest)
    return feet, legs


def extra_features(scene, robot):
    """Robot-specific prior, absolute tie-breaks and greedy/stair hypotheses."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'oracle'))
    import policy_v5
    prior = policy_v5.predict_all(scene)[robot]
    start = tuple(scene['robot']['rc'])
    heading = HEAD[scene['robot']['heading']]
    adj = adjacency(scene, robot == 4)
    pairs = {ACTIONS[(v[0]-start[0],v[1]-start[1])]: (v,e) for v,e in adj[start]}
    via, goal = locations(scene, 'via'), locations(scene, 'goal')
    targets = [v for v in via if v is not None] or [g for g in goal if g is not None]
    columns = [np.array([[float(a == prior), *[float(a==x) for x in range(4)],
                          *[float(heading==x) for x in range(4)]] for a in range(4)])]
    for metric in ['euclidean', 'manhattan', 'chebyshev']:
        for cp in [0., .5, 1., 2., 3., 5., 10.]:
            scores = np.full(4, np.inf)
            for action, (point, edge) in pairs.items():
                deltas = [(abs(point[0]-p[0]),abs(point[1]-p[1])) for p in targets]
                if deltas:
                    distances = [math.hypot(*d) if metric=='euclidean' else sum(d) if metric=='manhattan' else max(d) for d in deltas]
                    scores[action] = min(distances) + cp * (edge['status']=='crowded')
            columns.append(normalized_costs(scores))
    if robot == 4:
        for stairs in [-.5, .5, 1., 2., 5.]:
            columns.append(normalized_costs(route_scores(scene, True, (2, 1, 0, 0, 1, stairs))))
    return np.concatenate(columns, axis=1).astype(np.float32)


def predict_model(model, features, legal):
    scores = model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1].reshape(-1, 4)
    scores[~legal] = -1
    return scores.argmax(1)


def fit(data, out, rounds):
    train, val = [], []
    for split, target in [('train', train), ('validation', val)]:
        target.extend(json.loads((data / split / 'scenes.json').read_text(encoding='utf8')))
    labels = np.array(json.loads((data / 'train/labels.json').read_text())).reshape(-1, 10)
    val_labels = np.array(json.loads((data / 'validation/labels.json').read_text())).reshape(-1, 10)
    trf = cached_features(train, out / 'train_features.npz')
    vf = cached_features(val, out / 'validation_features.npz')
    learn, hold = train_test_split(np.arange(len(train)), test_size=.2, random_state=2026)
    models, accs, hold_scores = [], [], []
    preds = np.zeros_like(val_labels)
    valid_column = 5 + 4 + 10 + 10 + len(REFS) * 2
    for robot in range(10):
        x = np.concatenate([trf[robot == 4], np.array([extra_features(s, robot) for s in train])], axis=2)
        v = np.concatenate([vf[robot == 4], np.array([extra_features(s, robot) for s in val])], axis=2)
        target = (np.arange(4)[None, :] == labels[:, robot, None]).astype(int)
        model = XGBClassifier(n_estimators=rounds, max_depth=4, learning_rate=.04,
                              subsample=.9, colsample_bytree=.9, reg_lambda=8,
                              min_child_weight=6, n_jobs=4, random_state=2026,
                              objective='binary:logistic', eval_metric='logloss')
        model.fit(x[learn].reshape(-1, x.shape[-1]), target[learn].ravel())
        hold_pred = predict_model(model, x[hold], x[hold, :, valid_column].astype(bool))
        hold_score = float(np.mean(hold_pred == labels[hold, robot]))
        model.fit(x.reshape(-1, x.shape[-1]), target.ravel())
        preds[:, robot] = predict_model(model, v, v[:, :, valid_column].astype(bool))
        acc = float(np.mean(preds[:, robot] == val_labels[:, robot]))
        hold_scores.append(hold_score); accs.append(acc); models.append(model)
        print(f'R{robot}: train-holdout {hold_score:.4f}, validation {acc:.4f}', flush=True)
    bundle = {'version': VERSION, 'bank': BANK, 'models': models, 'valid_column': valid_column}
    joblib.dump(bundle, out / 'policy.joblib')
    report = {'macro': float(np.mean(accs)), 'per_robot': accs, 'train_holdout': hold_scores}
    (out / 'policy_metrics.json').write_text(json.dumps(report, indent=2))
    (out / 'oracle_validation_predictions.json').write_text(json.dumps(preds.ravel().tolist()))
    print(json.dumps(report), flush=True)


class Policy:
    def __init__(self, path):
        self.bundle = joblib.load(path)
        if self.bundle['version'] != VERSION or self.bundle['bank'] != BANK:
            raise ValueError('Policy feature/version mismatch; refit required')

    def predict_all(self, scene):
        x = [scene_features(scene), scene_features(scene, True)]
        result = {}
        for robot, model in enumerate(self.bundle['models']):
            features = np.concatenate([x[robot == 4], extra_features(scene, robot)], axis=1)[None]
            legal = features[:, :, self.bundle['valid_column']].astype(bool)
            result[robot] = int(predict_model(model, features, legal)[0])
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('artifacts/private'))
    p.add_argument('--rounds', type=int, default=300)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fit(a.data, a.out, a.rounds)
