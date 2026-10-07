"""Infer test ref (gref/vref kind+anchor) bang PhoBERT."""
import json, pathlib, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sys
sys.path.insert(0,'/mnt/hdd2/qtech/Phenika-AI/solution/nlp')
from train_phobert_full import MT, DS, coll, LABELS, REFS, BASE
# MT hien chi co g/v/u/f/r(gref). Can them vref+anchor? Tam dung gref + heuristic anchor tu text?
# Don gian: infer gref kind cho test (7 lop), anchor se suy tu text bang rule (ten dia diem gan nhat khac goal?) - se cai thien sau.
# De nhanh len 99%: truoc mat dung gref + spatial resolver (north/south/east/west khong can anchor, near/far can anchor -> fallback gan nhat).
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok=AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
te_obs=json.loads((BASE/'test/observations.json').read_text(encoding='utf-8'))
texts=[]
for o in te_obs:
    if not texts or texts[-1]!=o['mission']: texts.append(o['mission'])
# dedup giu order? test 1200 scenes, moi scene 1 text (10 rows lap). Lay moi 10:
texts=[te_obs[i]['mission'] for i in range(0,len(te_obs),10)]
print(len(texts))
m=MT().to(dev); m.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/phobert_mt.pt',map_location=dev)); m.eval()
# DS can scenes (co labels) - tao fake scenes chi de encode? Dung truc tiep tokenizer:
outs=[]
with torch.no_grad():
    for i in range(0,len(texts),32):
        batch=texts[i:i+32]
        enc=tok(batch,padding=True,truncation=True,max_length=128,return_tensors='pt')
        enc={k:v.to(dev) for k,v in enc.items()}
        _,_,_,_,pr=m(enc)
        for j in range(len(batch)):
            outs.append(REFS[pr[j].argmax().item()])
from collections import Counter
print(Counter(outs))
pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/test_gref_pred.json').write_text(json.dumps(outs))
print('saved test_gref')
