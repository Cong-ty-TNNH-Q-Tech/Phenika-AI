"""Infer PhoBERT test missions -> goal/via/urg/frag."""
import json, pathlib, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sys
sys.path.insert(0,'/mnt/hdd2/qtech/Phenika-AI/solution/nlp')
from train_phobert_full import MT, DS, coll, LABELS, REFS, BASE
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok=AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
te_obs=json.loads((BASE/'test/observations.json').read_text(encoding='utf-8'))
# dedup scenes (10 rows chung text)
texts=[]; seen=set()
for o in te_obs:
    if o['mission'] not in seen:
        seen.add(o['mission']); texts.append(o['mission'])
print(f'test unique missions: {len(texts)} / rows {len(te_obs)}')
# fake scenes for DS (need mission dict; chi can text de encode, labels dummy)
# encode truc tiep
m=MT().to(dev)
m.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/phobert_mt.pt',map_location=dev))
m.eval()
INV_L={i:l for i,l in enumerate(LABELS)}
outs=[]
with torch.no_grad():
    for i in range(0,len(texts),32):
        batch=texts[i:i+32]
        enc=tok(batch,padding=True,truncation=True,max_length=128,return_tensors='pt')
        enc={k:v.to(dev) for k,v in enc.items()}
        pg,pv,pu,pf,pr=m(enc)
        for j in range(len(batch)):
            outs.append({'text':batch[j],'goal':INV_L[pg[j].argmax().item()],'via':(INV_L[pv[j].argmax().item()] if pv[j].argmax().item()<10 else None),'urgent':bool(pu[j].argmax().item()),'fragile':bool(pf[j].argmax().item())})
print(f'inferred {len(outs)}')
pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/test_missions_pred.json').write_text(json.dumps(outs,ensure_ascii=False,indent=1),encoding='utf-8')
print('saved solution/nlp/test_missions_pred.json')
from collections import Counter
print('goal dist:',Counter(o['goal'] for o in outs))
print('urg:',sum(o['urgent'] for o in outs),'frag:',sum(o['fragile'] for o in outs))
