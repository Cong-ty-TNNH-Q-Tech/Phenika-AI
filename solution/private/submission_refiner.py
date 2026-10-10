"""Train-only correction of an existing automatic submission.

Policy-generated train predictions are the training inputs. Saved CV validation
predictions assess whether that transfer improves the real pipeline. Model and
per-robot selection are frozen before loading test at inference.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

from policy_ranker import TYPES, STYLES, REFS, predict_model
from nlp_fast import flatten
from mission_flags import apply as apply_flags
from style_fast import features as style_features

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'solution/oracle'))
sys.path.insert(0, str(ROOT / 'solution/nlp'))
import policy_v5
from nlp_rules import rule_kind


def features(preds, missions, styles, robot):
    # Candidate comparisons are invariant to rotations of absolute actions.
    rows = []
    for predictions, mission, style in zip(preds, missions, styles):
        context = [bool(mission['urgent']), bool(mission['fragile']), mission['via'] is not None]
        context += [mission['goal'] == x for x in TYPES]
        context += [mission['via'] == x for x in TYPES]
        context += [mission.get('gref_kind', 'NONE') == x for x in REFS]
        context += [style == x for x in STYLES]
        votes = np.bincount(predictions, minlength=4)
        candidate_rows = []
        for action in range(4):
            same = [int(p == action) for p in predictions]
            opposite = {0:1,1:0,2:3,3:2}[action]
            opposite_votes = [int(p == opposite) for p in predictions]
            candidate_rows.append(context + same + opposite_votes +
                                  [same[robot], votes[action]/10., votes[opposite]/10.,
                                   len(set(predictions))/4.])
        rows.append(candidate_rows)
    return np.array(rows, dtype=np.float32)


def predicted_missions(project, split, observations):
    missions = json.loads((project / f'solution/nlp/v3_{split}_missions.json').read_text())
    for i, mission in enumerate(missions):
        kind = rule_kind(observations[i*10]['mission'])
        if kind is not None:
            mission['gref_kind'] = kind
        missions[i] = apply_flags(observations[i*10]['mission'], mission)
    return missions


def train(data, out, project):
    scenes = json.loads((data/'train/scenes.json').read_text(encoding='utf8'))
    labels = np.array(json.loads((data/'train/labels.json').read_text())).reshape(-1,10)
    val_scenes = json.loads((data/'validation/scenes.json').read_text(encoding='utf8'))
    val_labels = np.array(json.loads((data/'validation/labels.json').read_text())).reshape(-1,10)
    train_pred = np.array([[p[r] for r in range(10)] for p in map(policy_v5.predict_all, scenes)])
    # Teacher can fail to find a route in an annotated scene; don't invent a label.
    train_pred = np.array([[2 if p is None else p for p in row] for row in train_pred],dtype=int)
    base = np.array(json.loads((project/'solution/predictions_v5_val.json').read_text())).reshape(-1,10)
    observations = json.loads((data/'validation/observations.json').read_text(encoding='utf8'))
    mission_train = [flatten(s['mission']) for s in scenes]
    mission_val = predicted_missions(project,'validation',observations)
    # Use actual image style inference also on validation (not its annotation).
    from PIL import Image
    style_model=joblib.load(out/'style.joblib')
    val_style = style_model.predict([style_features(Image.open(data/'validation'/s['image'])) for s in val_scenes])
    scores, choices, models = [], [], []
    candidate = base.copy()
    for r in range(10):
        x=features(train_pred,mission_train,[s['style'] for s in scenes],r)
        v=features(base,mission_val,val_style,r)
        y=(np.arange(4)[None,:]==labels[:,r,None]).astype(int)
        model=XGBClassifier(n_estimators=180,max_depth=3,learning_rate=.04,
                            reg_lambda=12,min_child_weight=12,n_jobs=4,random_state=2026,
                            subsample=.9,colsample_bytree=.9,objective='binary:logistic')
        model.fit(x.reshape(-1,x.shape[-1]),y.ravel())
        # Candidate must have occurred in the existing non-legged predictions;
        # R4 may retain its own unique stairs-capable candidate.
        legal=np.array([[a in [p[j] for j in range(10) if j!=4] or (r==4 and a==p[4])
                         for a in range(4)] for p in base])
        candidate[:,r]=predict_model(model,v,legal)
        before=float(np.mean(base[:,r]==val_labels[:,r]))
        after=float(np.mean(candidate[:,r]==val_labels[:,r]))
        choices.append(after-before >= .01)
        scores.append({'robot':r,'baseline':before,'candidate':after})
        models.append(model)
        print(f'refiner R{r}: {before:.4f} -> {after:.4f}',flush=True)
    refined=np.where(np.array(choices)[None,:],candidate,base)
    delta=(refined==val_labels).mean(1)-(base==val_labels).mean(1)
    rng=np.random.default_rng(2026)
    ci=np.quantile(delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1),[.025,.975])
    report={'baseline_macro':float((base==val_labels).mean()),
            'selected_macro':float((refined==val_labels).mean()),'use_refiner':choices,
            'per_robot':scores,'paired_delta_interval_95':ci.tolist(),
            'warning':'Cross-robot candidates are not guaranteed legal without an inferred graph.'}
    joblib.dump({'models':models,'use_refiner':choices},out/'refiner.joblib')
    (out/'refiner_metrics.json').write_text(json.dumps(report,indent=2))
    (out/'refined_validation_predictions.json').write_text(json.dumps(refined.ravel().tolist()))
    print(json.dumps(report),flush=True)


def infer(data,out,project):
    bundle=joblib.load(out/'refiner.joblib')
    report=json.loads((out/'refiner_metrics.json').read_text())
    if report['selected_macro'] <= report['baseline_macro'] or report['paired_delta_interval_95'][0] <= 0:
        raise ValueError('Refiner did not demonstrate a robust validation gain; preserve existing submission')
    # Only now open test inputs to run frozen inference. No rule fitting here.
    obs=json.loads((data/'test/observations.json').read_text(encoding='utf8'))
    base=np.array(json.loads((project/'solution/predictions.json').read_text())).reshape(-1,10)
    if len(obs)!=len(base)*10 or any(obs[i]['robot_id']!=i%10 for i in range(len(obs))):
        raise ValueError('Test row ordering/length mismatch')
    missions=predicted_missions(project,'test',obs)
    from PIL import Image
    style_model=joblib.load(out/'style.joblib')
    styles=style_model.predict([style_features(Image.open(data/'test'/obs[i*10]['image'])) for i in range(len(base))])
    result=base.copy()
    for r,model in enumerate(bundle['models']):
        if not bundle['use_refiner'][r]:continue
        x=features(base,missions,styles,r)
        legal=np.array([[a in [p[j] for j in range(10) if j!=4] or (r==4 and a==p[4])
                         for a in range(4)] for p in base])
        result[:,r]=predict_model(model,x,legal)
    path=out/'predictions_refined_test.json'
    path.write_text(json.dumps(result.ravel().tolist()))
    print('saved',path,'changes',int((result!=base).sum()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['train','infer'])
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('artifacts/private'))
    p.add_argument('--project',type=Path,default=ROOT)
    a=p.parse_args()
    (train if a.mode=='train' else infer)(a.data,a.out,a.project)
