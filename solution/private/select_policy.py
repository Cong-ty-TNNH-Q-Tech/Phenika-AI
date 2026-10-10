"""Select per-robot policies using CV validation graphs; never load test."""
import argparse
import json
from pathlib import Path

import numpy as np


def choose(graphs_path, labels_path, baseline_path, out):
    graphs = json.loads(graphs_path.read_text())
    labels = np.array(json.loads(labels_path.read_text())).reshape(-1, 10)
    base = np.array(json.loads(baseline_path.read_text())).reshape(-1, 10)
    ranker = base.copy()
    rebuilt = base.copy()
    seen = set()
    for graph in graphs:
        i = graph['_scene_index']
        if i in seen or not 0 <= i < len(labels):
            raise ValueError('Duplicate/out-of-range scene index in capture')
        seen.add(i)
        ranker[i] = graph['_ranker']
        rebuilt[i] = graph['_baseline']
    baseline_scores = (rebuilt == labels).mean(0)
    ranker_scores = (ranker == labels).mean(0)
    # Conservative minimum gain: do not switch on a single lucky validation row.
    use = (ranker_scores - baseline_scores >= .01)
    combined = np.where(use[None, :], ranker, rebuilt)
    delta = (combined == labels).mean(1) - (rebuilt == labels).mean(1)
    rng = np.random.default_rng(2026)
    boot = np.mean(delta[rng.integers(len(delta), size=(2000, len(delta)))], axis=1)
    report = {'use_ranker': use.tolist(), 'baseline_macro': float((rebuilt==labels).mean()),
              'ranker_macro': float((ranker==labels).mean()), 'selected_macro': float((combined==labels).mean()),
              'baseline_per_robot': baseline_scores.tolist(), 'ranker_per_robot': ranker_scores.tolist(),
              'paired_scene_delta_interval_95': np.quantile(boot,[.025,.975]).tolist(),
              'captured_scenes': len(seen), 'total_scenes': len(labels)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    out.with_name('predictions_selected_validation.json').write_text(json.dumps(combined.ravel().tolist()))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    choose(a.graphs,a.labels,a.baseline,a.out)
