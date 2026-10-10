"""Train a compact style classifier from train images only."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier


def features(im):
    rgb = np.asarray(im.convert('RGB').resize((48, 48)), dtype=float).reshape(-1, 3) / 255
    return np.concatenate([rgb.mean(0), rgb.std(0),
                           np.quantile(rgb, [.1, .25, .5, .75, .9], axis=0).ravel(),
                           [(rgb.mean(1) < .3).mean(), (rgb.mean(1) > .9).mean()],
                           np.histogram(rgb[:, 0] - rgb[:, 2], bins=12, range=(-1, 1))[0] / len(rgb)])


def main(data, out):
    arrays, targets = {}, {}
    for split in ['train', 'validation']:
        scenes = json.loads((data / split / 'scenes.json').read_text(encoding='utf8'))
        arrays[split] = [features(Image.open(data / split / s['image'])) for s in scenes]
        targets[split] = [s['style'] for s in scenes]
    model = ExtraTreesClassifier(n_estimators=80, max_depth=10, n_jobs=4, random_state=2026)
    model.fit(arrays['train'], targets['train'])
    score = float(np.mean(model.predict(arrays['validation']) == targets['validation']))
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / 'style.joblib')
    (out / 'style_metrics.json').write_text(json.dumps({'validation_accuracy': score}))
    print('style validation', score, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('artifacts/private'))
    a = p.parse_args()
    main(a.data, a.out)
