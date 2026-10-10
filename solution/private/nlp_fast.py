"""CPU-trainable mission parser, including both goal and via references."""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MAP_KINDS = {'north_most', 'south_most', 'west_most', 'east_most', 'anchor_near'}
FIELDS = ['goal', 'via', 'urgent', 'fragile', 'gref_kind', 'gref_anchor',
          'via_ref_kind', 'via_ref_anchor']


def normalize(text):
    text = ''.join(x for x in unicodedata.normalize('NFD', text.lower())
                   if unicodedata.category(x) != 'Mn').replace('đ', 'd')
    # Remove explicitly cancelled/history clauses, preserving corrected instructions.
    text = re.sub(r'\([^)]*(?:nham|ghi)[^)]*\)', ' ', text)
    text = re.sub(r'(?:hom qua da giao|nguoi nhan da roi|dung nham voi|khong can ghe|'
                  r'luc nay nhan nham|huy don giao)[^.!?:]*[.!?]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def flatten(m):
    result = {k: m[k] for k in ['goal', 'via', 'urgent', 'fragile']}
    result.update(gref_kind=(m['goal_ref'] or {}).get('kind', 'NONE'),
                  gref_anchor=(m['goal_ref'] or {}).get('anchor'),
                  via_ref_kind=(m['via_ref'] or {}).get('kind', 'NONE'),
                  via_ref_anchor=(m['via_ref'] or {}).get('anchor'))
    return result


def encode(value):
    return 'NONE' if value is None else str(value)


class Parser:
    def __init__(self, path):
        self.bundle = joblib.load(path)

    def predict(self, texts):
        features = self.bundle['vectorizer'].transform([normalize(t) for t in texts])
        outputs = [{} for _ in texts]
        for field, model in self.bundle['models'].items():
            pred = model.predict(features) if hasattr(model, 'predict') else [model] * len(texts)
            for i, value in enumerate(pred):
                if field in ['urgent', 'fragile']:
                    value = value == 'True'
                elif field in ['goal', 'via', 'gref_anchor', 'via_ref_anchor'] and value == 'NONE':
                    value = None
                outputs[i][field] = value
        return outputs


def train(data, out):
    train = json.loads((data / 'train/scenes.json').read_text(encoding='utf8'))
    val = json.loads((data / 'validation/scenes.json').read_text(encoding='utf8'))
    texts = [normalize(s['mission']['text']) for s in train]
    flat = [flatten(s['mission']) for s in train]
    vflat = [flatten(s['mission']) for s in val]
    learn, hold = train_test_split(np.arange(len(train)), test_size=.2, random_state=2026)
    def vectorizer():
        return FeatureUnion([
            ('char', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2,
                                     max_features=60000, sublinear_tf=True)),
            ('word', TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=40000,
                                     sublinear_tf=True))])
    vector = vectorizer()
    vector.fit([texts[i] for i in learn])
    x, h = vector.transform([texts[i] for i in learn]), vector.transform([texts[i] for i in hold])
    hold_metrics = {}
    for field in FIELDS:
        # The kind of an unnamed map goal cannot be learned from text.
        indices = [j for j, i in enumerate(learn) if field != 'goal' or flat[i]['gref_kind'] not in MAP_KINDS]
        target = [encode(flat[learn[j]][field]) for j in indices]
        model = LogisticRegression(C=15, max_iter=300, solver='liblinear') if len(set(target)) < 3 else LogisticRegression(C=15, max_iter=300, solver='lbfgs')
        model.fit(x[indices], target)
        ids = [j for j, i in enumerate(hold) if field != 'goal' or flat[i]['gref_kind'] not in MAP_KINDS]
        hold_metrics[field] = float(np.mean(model.predict(h[ids]) == [encode(flat[hold[j]][field]) for j in ids]))
        print(f'{field}: train holdout {hold_metrics[field]:.3f}', flush=True)
    vector = vectorizer(); x = vector.fit_transform(texts)
    models = {}
    for field in FIELDS:
        indices = [i for i in range(len(train)) if field != 'goal' or flat[i]['gref_kind'] not in MAP_KINDS]
        target = [encode(flat[i][field]) for i in indices]
        model = LogisticRegression(C=15, max_iter=350, solver='lbfgs')
        model.fit(x[indices], target); models[field] = model
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({'vectorizer': vector, 'models': models}, out / 'nlp.joblib')
    outputs = Parser(out / 'nlp.joblib').predict([s['mission']['text'] for s in val])
    metrics = {}
    for field in FIELDS:
        ids = [i for i in range(len(val)) if field != 'goal' or vflat[i]['gref_kind'] not in MAP_KINDS]
        metrics[field] = float(np.mean([outputs[i][field] == vflat[i][field] for i in ids]))
    (out / 'nlp_metrics.json').write_text(json.dumps({'train_holdout': hold_metrics, 'validation': metrics}, indent=2))
    (out / 'nlp_validation_missions.json').write_text(json.dumps(outputs))
    print('validation', metrics, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('artifacts/private'))
    a = p.parse_args()
    train(a.data, a.out)
