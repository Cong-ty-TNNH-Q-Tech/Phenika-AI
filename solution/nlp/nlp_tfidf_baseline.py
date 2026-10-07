"""Vong 3a: NLP TF-IDF baseline."""
import json, pathlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
tr_s=json.loads((BASE/'train/scenes.json').read_text(encoding='utf-8'))
va_s=json.loads((BASE/'validation/scenes.json').read_text(encoding='utf-8'))
def prep(scenes):
    X=[s['mission']['text'] for s in scenes]
    y_goal=[s['mission']['goal'] for s in scenes]
    y_via=[s['mission']['via'] if s['mission']['via'] else 'NONE' for s in scenes]
    y_urg=[int(s['mission']['urgent']) for s in scenes]
    y_frag=[int(s['mission']['fragile']) for s in scenes]
    def ref_kind(r):
        return r['kind'] if r else 'NONE'
    y_gref=[ref_kind(s['mission']['goal_ref']) for s in scenes]
    y_vref=[ref_kind(s['mission']['via_ref']) for s in scenes]
    return X,y_goal,y_via,y_urg,y_frag,y_gref,y_vref
Xtr,*ytr=prep(tr_s); Xva,*yva=prep(va_s)
names=['goal(10)','via(11)','urgent','fragile','goal_ref','via_ref']
vec=TfidfVectorizer(max_features=8000, ngram_range=(1,2))
Xtr_v=vec.fit_transform(Xtr); Xva_v=vec.transform(Xva)
for i,name in enumerate(names):
    clf=LogisticRegression(max_iter=1000, C=4)
    clf.fit(Xtr_v, ytr[i])
    acc=accuracy_score(yva[i], clf.predict(Xva_v))
    print(f'{name}: val acc={acc:.4f}')
    # train acc
    print(f'  train acc={accuracy_score(ytr[i], clf.predict(Xtr_v)):.4f}')
