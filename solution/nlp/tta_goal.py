"""TTA goal: vote original/strip/lower voi PhoBERT 10ep. Do val."""
import json, pathlib, unicodedata, collections
import torch
from transformers import AutoTokenizer
import sys
sys.path.insert(0,'/mnt/hdd2/qtech/Phenika-AI/solution/nlp')
from train_phobert_full import MT, LABELS, BASE
def strip(s): return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok=AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
va=json.loads((BASE/'validation/scenes.json').read_text(encoding='utf-8'))
m=MT().to(dev); m.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/phobert_mt.pt',map_location=dev)); m.eval()
hit1=hit3=0
with torch.no_grad():
    for s in va:
        t=s['mission']['text']; y=LABELS.index(s['mission']['goal'])
        variants=[t,strip(t),t.lower()]
        preds=[]
        for v in variants:
            enc=tok(v,return_tensors='pt',truncation=True,max_length=128)
            enc={k:x.to(dev) for k,x in enc.items()}
            pg,_,_,_,_=m(enc)
            preds.append(pg.argmax(1).item())
        if preds[0]==y: hit1+=1
        # majority vote 3
        vote=collections.Counter(preds).most_common(1)[0][0]
        if vote==y: hit3+=1
print(f'val goal single={hit1}/{len(va)}={hit1/len(va):.4f}, TTA3-vote={hit3}/{len(va)}={hit3/len(va):.4f}')
