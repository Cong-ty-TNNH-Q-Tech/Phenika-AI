"""Mine anchor phrases from train only; no inspection of test text."""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

from nlp_fast import normalize

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'solution/nlp'))
import nlp_rules

PATTERN=re.compile(r'\b(?:gan|xa|sat|canh|ke)\s+(.{2,75}?)(?:\s+(?:nhat|hon)\b|[.!?;,])')


def mine(data,out):
    scenes=json.loads((data/'train/scenes.json').read_text(encoding='utf8'))
    phrases=collections.defaultdict(collections.Counter)
    for s in scenes:
        m=s['mission'];g=m['goal_ref'] or {};v=m['via_ref'] or {}
        anchor=g.get('anchor')
        if not anchor or (v.get('anchor') is not None and v['anchor']!=anchor):continue
        for match in PATTERN.finditer(normalize(m['text'])):
            phrase=match.group(1).strip()
            if 1<=len(phrase.split())<=7 and not any(x in phrase for x in [' lay ',' truoc ','roi ','giao ']):
                phrases[phrase][anchor]+=1
    result={}
    for phrase,counts in phrases.items():
        typ,count=counts.most_common(1)[0]
        if count>=2 and count/sum(counts.values())>=.9:
            result.setdefault(typ,{})[phrase]=float(10+count)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('train mined aliases',sum(len(x) for x in result.values()))


def install(path):
    d=json.loads(path.read_text(encoding='utf8'))
    for typ,aliases in d.items():
        nlp_rules.ALIAS.setdefault(typ,{}).update(aliases)


def predict(text,mission,landmarks):
    goal=None if mission.get('gref_kind')=='anchor_near' else mission['goal']
    return nlp_rules.match_anchor(normalize(text),goal,mission['via'],landmarks)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('artifacts/private/anchor_aliases.json'))
    a=p.parse_args();mine(a.data,a.out)
