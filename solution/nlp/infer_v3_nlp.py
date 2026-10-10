"""Infer PhoBERT v3 NLP -> mission preds (goal/via/urg/frag/gref_kind/gref_anchor)."""
import json, pathlib, torch, sys
from transformers import AutoTokenizer
sys.path.insert(0,'/mnt/hdd2/qtech/Phenika-AI/solution/nlp')
from train_phobert_v3 import MT, LABELS, REFS, VREFS
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok=AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
def infer(texts, model_path):
    m=MT().to(dev); m.load_state_dict(torch.load(model_path,map_location=dev)); m.eval()
    outs=[]
    with torch.no_grad():
        for i in range(0,len(texts),32):
            b=texts[i:i+32]
            enc=tok(b,padding=True,truncation=True,max_length=160,return_tensors='pt'); enc={k:v.to(dev) for k,v in enc.items()}
            pg,pv,pu,pf,pr,pra,pvr,pva=m(enc)
            for j in range(len(b)):
                vi=pv[j].argmax().item(); ri=pr[j].argmax().item(); ai=pra[j].argmax().item()
                outs.append({'goal':LABELS[pg[j].argmax().item()],'via':(LABELS[vi] if vi<10 else None),
                    'urgent':bool(pu[j].argmax().item()),'fragile':bool(pf[j].argmax().item()),
                    'gref_kind':REFS[ri],'gref_anchor':(LABELS[ai] if ai<10 else None)})
    return outs
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--split',default='validation'); ap.add_argument('--out',default=None)
    a=ap.parse_args()
    BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/v3data/delivery_public')
    obs=json.loads((BASE/f'{a.split}/observations.json').read_text(encoding='utf-8'))
    texts=[obs[i]['mission'] for i in range(0,len(obs),10)]
    outs=infer(texts,'/mnt/hdd2/qtech/Phenika-AI/solution/nlp/phobert_v3.pt')
    out=a.out or f'/mnt/hdd2/qtech/Phenika-AI/solution/nlp/v3_{a.split}_missions.json'
    pathlib.Path(out).write_text(json.dumps(outs,ensure_ascii=False))
    from collections import Counter
    print(f'saved {out} ({len(outs)})')
    print('gref_kind:',Counter(o['gref_kind'] for o in outs))
