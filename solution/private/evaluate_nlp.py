"""Measure NLP changes with true val graph; this is not end-to-end accuracy."""
import json
import sys
from pathlib import Path

import numpy as np
from nlp_fast import normalize
from references import apply_mission
from mission_flags import apply as flags
import anchor_miner

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'solution/oracle'))
sys.path.insert(0,str(ROOT/'solution/nlp'))
import policy_v5
import nlp_rules


def main(data,out):
    scenes=json.loads((data/'validation/scenes.json').read_text(encoding='utf8'))
    labels=np.array(json.loads((data/'validation/labels.json').read_text())).reshape(-1,10)
    missions=json.loads((ROOT/'solution/nlp/v3_validation_missions.json').read_text())
    rows=[]
    for s,m in zip(scenes,missions):
        p=dict(m)
        p['gref_kind']=nlp_rules.rule_kind(s['mission']['text']) or p['gref_kind']
        if p['gref_kind'] in ['near','far','anchor_near']:
            p['gref_anchor']=nlp_rules.match_anchor(s['mission']['text'],p['goal'],p['via'],s['landmarks']) or p['gref_anchor']
        graph=apply_mission(s,p)
        pred=policy_v5.predict_all(graph)
        rows.append([pred[r] for r in range(10)])
    before=float((np.array(rows)==labels).mean())
    anchor_miner.install(out/'anchor_aliases.json')
    rows=[]
    for s,m in zip(scenes,missions):
        p=flags(s['mission']['text'],m)
        p['gref_kind']=nlp_rules.rule_kind(normalize(s['mission']['text'])) or p['gref_kind']
        if p['gref_kind'] in ['near','far','anchor_near']:
            p['gref_anchor']=anchor_miner.predict(s['mission']['text'],p,s['landmarks']) or p['gref_anchor']
        graph=apply_mission(s,p)
        pred=policy_v5.predict_all(graph)
        rows.append([pred[r] for r in range(10)])
    after=float((np.array(rows)==labels).mean())
    report={'true_graph_old_nlp_macro':before,'true_graph_improved_nlp_macro':after,
            'per_robot':(np.array(rows)==labels).mean(0).tolist()}
    (out/'nlp_ablation_metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('artifacts/private'))
    a=p.parse_args();main(a.data,a.out)
