"""Structured oracle benchmark for Courier v3.

This script intentionally consumes only train/validation ``scenes.json``.  It is
for reverse-engineering robot policy with perfect graph/mission annotations;
it never inspects test while fitting features or rules.
"""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier


ACTIONS = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
STATUS = ("normal", "crowded", "covered")
WEIGHTS = (
    (1.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 5.0, 1.0),
    (1.0, 1.0, .15), (1.0, 1.0, .5), (1.0, 1.0, 2.0),
    (1.0, 3.0, .3), (1.0, 8.0, .2),
)
HEAD = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}


def adjacency(scene, legged):
    out = collections.defaultdict(list)
    for edge in scene["edges"]:
        if edge["status"] == "closed" or (edge["stairs"] and not legged):
            continue
        a, b = tuple(edge["a"]), tuple(edge["b"])
        target = edge["oneway_to"]
        if target is None:
            out[a].append((b, edge)); out[b].append((a, edge))
        elif tuple(target) == b:
            out[a].append((b, edge))
        elif tuple(target) == a:
            out[b].append((a, edge))
    return out


def locations(scene, key):
    mission = scene["mission"]
    kind, ref = mission[key], mission[f"{key}_ref"]
    if kind is None:
        return [None]
    if ref is not None:
        return [tuple(ref["rc"])]
    return [tuple(x["rc"]) for x in scene["landmarks"] if x["type"] == kind]


def shortest(adj, start, target, weights):
    """Dijkstra distance, and per-status counts on its deterministic path."""
    import heapq
    queue = [(0., (0, 0, 0), start)]
    seen = set()
    while queue:
        dist, counts, node = heapq.heappop(queue)
        if node in seen:
            continue
        seen.add(node)
        if node == target:
            return dist, counts
        for nxt, edge in adj[node]:
            if nxt in seen:
                continue
            idx = STATUS.index(edge["status"]) if edge["status"] in STATUS else 0
            plus = list(counts); plus[idx] += 1
            heapq.heappush(queue, (dist + weights[idx], tuple(plus), nxt))
    return math.inf, (0, 0, 0)


def route_cost(adj, first, via, goal, weights):
    """Cost after forcing one initial edge, optimized over duplicate landmarks."""
    nxt, edge = first
    first_status = STATUS.index(edge["status"]) if edge["status"] in STATUS else 0
    best = (math.inf, (0, 0, 0))
    for waypoint in via:
        for target in goal:
            if target is None:
                continue
            if waypoint is None:
                dist, counts = shortest(adj, nxt, target, weights)
            else:
                d1, c1 = shortest(adj, nxt, waypoint, weights)
                d2, c2 = shortest(adj, waypoint, target, weights)
                dist, counts = d1 + d2, tuple(x + y for x, y in zip(c1, c2))
            candidate = (weights[first_status] + dist,
                         tuple(counts[i] + (i == first_status) for i in range(3)))
            if candidate[0] < best[0]:
                best = candidate
    return best


def relative(heading, action):
    if action == heading:
        return 0
    if (heading, action) in ((0, 3), (3, 1), (1, 2), (2, 0)):
        return 1
    if (heading, action) in ((0, 2), (2, 1), (1, 3), (3, 0)):
        return 2
    return 3


def candidate_features(scene, robot_id):
    """Four rows (one/action), all constructed from the annotated train/val graph."""
    adj = adjacency(scene, robot_id == 4)
    start = tuple(scene["robot"]["rc"])
    heading = HEAD[scene["robot"]["heading"]]
    via, goal = locations(scene, "via"), locations(scene, "goal")
    mission = scene["mission"]
    ref = mission["goal_ref"] or {}
    base = [
        float(mission["urgent"]), float(mission["fragile"]),
        float(scene["weather"] == "rain"),
        *[float(scene["style"] == x) for x in ("classic", "night", "print", "sketch")],
        float(mission["via"] is not None), len(via), len(goal),
        *[float(ref.get("kind") == x) for x in
          ("north_most", "south_most", "west_most", "east_most", "anchor_near", "near", "far")],
        len(scene["landmarks"]), len(scene["edges"]),
    ]
    rows = []
    for action in range(4):
        first = next(((v, e) for v, e in adj[start]
                      if ACTIONS[(v[0] - start[0], v[1] - start[1])] == action), None)
        valid = float(first is not None)
        feat = base + [valid] + [float(relative(heading, action) == x) for x in range(4)]
        if first is None:
            rows.append(feat + [99.] * (len(WEIGHTS) * 4 + 8))
            continue
        nxt, edge = first
        status = [float(edge["status"] == x) for x in STATUS]
        suffix = status + [float(edge["stairs"]), float(edge["oneway_to"] is not None)]
        for weights in WEIGHTS:
            distance, counts = route_cost(adj, first, via, goal, weights)
            suffix.extend([min(distance, 99.), *counts])
        # Geometry to every candidate target, useful when the robot is greedy.
        targets = [x for x in via if x is not None] if mission["via"] else [x for x in goal if x is not None]
        if targets:
            d = [abs(nxt[0] - x[0]) + abs(nxt[1] - x[1]) for x in targets]
            suffix.extend([min(d), max(d), np.mean(d)])
        else:
            suffix.extend([99., 99., 99.])
        rows.append(feat + suffix)
    return rows


def load(data, split):
    root = Path(data) / split
    return (json.loads((root / "scenes.json").read_text(encoding="utf-8")),
            json.loads((root / "labels.json").read_text(encoding="utf-8")))


def fit_predict(data, rounds=250):
    train, train_y = load(data, "train")
    val, val_y = load(data, "validation")
    pred = np.zeros((len(val), 10), dtype=np.int64)
    for robot in range(10):
        x_train = np.asarray([candidate_features(s, robot) for s in train], dtype=np.float32)
        x_val = np.asarray([candidate_features(s, robot) for s in val], dtype=np.float32)
        model = XGBClassifier(
            n_estimators=rounds, max_depth=6, learning_rate=.045, subsample=.85,
            colsample_bytree=.9, objective="multi:softprob", num_class=4,
            eval_metric="mlogloss", n_jobs=-1, random_state=2026,
        )
        model.fit(x_train.reshape(len(train), -1), train_y[robot::10])
        pred[:, robot] = model.predict(x_val.reshape(len(val), -1))
    scores = [(pred[:, r] == val_y[r::10]).mean() for r in range(10)]
    print("oracle structured macro=%.4f" % np.mean(scores))
    print("per robot", " ".join("R%d=%.3f" % (r, x) for r, x in enumerate(scores)))
    return pred, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=250)
    args = parser.parse_args()
    fit_predict(args.data, args.rounds)
