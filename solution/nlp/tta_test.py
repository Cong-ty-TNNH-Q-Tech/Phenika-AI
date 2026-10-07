"""TTA test goal/via/urg/frag: vote original/strip/lower."""
import json, pathlib, unicodedata, collections
import torch
from transformers import AutoTokenizer
import sys
sys.path.insert(0,'/mnt/hdd2/qtech/Phenika-AI/solution/nlp')
from train_phobert_full import MT, LABELS, BASE
def strip(s): return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok=AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
te_obs=json.loads((BASE/'test/observations.json').read_text(encoding='utf-8'))
texts=[te_obs[i]['mission'] for i in range(0,len(te_obs),10)]
m=MT().to(dev); m.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/phobert_mt.pt',map_location=dev)); m.eval()
outs=[]
with torch.no_grad():
    for i in range(0,len(texts),32):
        batch=texts[i:i+32]
        variants=[[t,strip(t),t.lower()] for t in batch]
        # vote per head? don gian vote goal, via lay mode, urg/frag OR?
        # chay 3 lan
        all_g=[]; all_v=[]; all_u=[]; all_f=[]
        for k in range(3):
            txts=[v[k] for v in variants]
            enc=tok(txts,padding=True,truncation=True,max_length=128,return_tensors='pt')
            enc={kk:vv.to(dev) for kk,vv in enc.items()}
            pg,pv,pu,pf,pr=m(enc)
            all_g.append(pg.argmax(1).cpu().tolist()); all_v.append(pv.argmax(1).cpu().tolist()); all_u.append(pu.argmax(1).cpu().tolist()); all_f.append(pf.argmax(1).cpu().tolist())
        for j in range(len(batch)):
            g=collections.Counter([all_g[k][j] for k in range(3)]).most_common(1)[0][0]
            v=collections.Counter([all_v[k][j] for k in range(3)]).most_common(1)[0][0]
            u=collections.Counter([all_u[k][j] for k in range(3)]).most_common(1)[0][0]
            f=collections.Counter([all_f[k][j] for k in range(3)]).most_common(1)[0][0]
            outs.append({'text':batch[j],'goal':LABELS[g],'via':(LABELS[v] if v<10 else None),'urgent':bool(u),'fragile':bool(f)})
pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/test_missions_tta.json').write_text(json.dumps(outs,ensure_ascii=False,indent=1),encoding='utf-8')
print(f'saved TTA {len(outs)}')
# so voi non-TTA?
old=json.loads(pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/test_missions_pred.json').read_text(encoding='utf-8'))
chg=sum(1 for a,b in zip(old,outs) if a['goal']!=b['goal'] or (a['via'] or '')!=(b['via'] or ''))
print(f'thay doi vs non-TTA: {chg}/{len(outs)}')
