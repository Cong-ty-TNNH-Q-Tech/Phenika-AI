"""Full pipeline v1: NLP (TF-IDF) + thong ke (robot, goal) -> du doan huong, khong can CV graph.
Dung de co predictions.json test ngay hom nay, lam baseline manh hon majority 26%."""
import json, pathlib, collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
# train goal predictor
tr_s=json.loads((BASE/'train/scenes.json').read_text(encoding='utf-8'))
va_s=json.loads((BASE/'validation/scenes.json').read_text(encoding='utf-8'))
tr_obs=json.loads((BASE/'train/observations.json').read_text())
tr_labels=json.loads((BASE/'train/labels.json').read_text())
va_obs=json.loads((BASE/'validation/observations.json').read_text())
va_labels=json.loads((BASE/'validation/labels.json').read_text())
te_obs=json.loads((BASE/'test/observations.json').read_text())
Xtr=[s['mission']['text'] for s in tr_s]
ygoal=[s['mission']['goal'] for s in tr_s]
vec=TfidfVectorizer(max_features=8000, ngram_range=(1,2))
Xtrv=vec.fit_transform(Xtr)
clf_goal=LogisticRegression(max_iter=1000,C=4)
clf_goal.fit(Xtrv, ygoal)
# hoc P(direction | robot, goal) tu train labels + true goal
from collections import Counter, defaultdict
stat=defaultdict(Counter)  # (robot, goal) -> Counter direction
# map scene -> goal
for i,s in enumerate(tr_s):
    g=s['mission']['goal']
    for r in range(10):
        y=tr_labels[i*10+r]
        stat[(r,g)][y]+=1
majority={k:c.most_common(1)[0][0] for k,c in stat.items()}
# global fallback per robot
fallback={}
for r in range(10):
    c=Counter(y for o,y in zip(tr_obs,tr_labels) if o['robot_id']==r)
    fallback[r]=c.most_common(1)[0][0]
def predict_goal(texts):
    return clf_goal.predict(vec.transform(texts))
# val eval
Xva=[s['mission']['text'] for s in va_s]
pgoals=predict_goal(Xva)
hit=defaultdict(int); tot=defaultdict(int)
for si,s in enumerate(va_s):
    pg=pgoals[si]
    for r in range(10):
        y=va_labels[si*10+r]
        p=majority.get((r,pg), fallback[r])
        tot[r]+=1
        if p==y: hit[r]+=1
macro=sum(hit[r]/tot[r] for r in range(10))/10
print(f'VAL (NLP goal + stat robot-goal, no CV): macro={macro:.4f}')
for r in range(10):
    print(f' R{r}:{hit[r]/tot[r]:.3f}',end=' ')
print()
# test infer: can mission text truc tiep tu observations (moi scene lap 10 lan)
# predict goal per scene (moi 10 dong chung text)
tes_texts=[]
for i in range(0,len(te_obs),10):
    tes_texts.append(te_obs[i]['mission'])
pte_goals=predict_goal(tes_texts)
preds=[]
for si,pg in enumerate(pte_goals):
    for r in range(10):
        preds.append(int(majority.get((r,pg), fallback[r])))
print(f'test preds: {len(preds)}')
pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/predictions_v1.json').write_text(json.dumps(preds))
print('saved solution/predictions_v1.json')
# so sanh voi baseline majority thuong
print('fallback majority:', fallback)
