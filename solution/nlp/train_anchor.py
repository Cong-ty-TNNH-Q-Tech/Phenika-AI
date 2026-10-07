"""Anchor classifier (11 lop: 10 types + NONE) cho near/far. Train nhanh."""
import json, pathlib, random, unicodedata
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TYPES=['library','dorm','sports','clinic','canteen','parking','lecture','lab','office','gate']
def strip(s): return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
class ADS(Dataset):
    def __init__(self,split):
        sc=json.loads((BASE/split/'scenes.json').read_text())
        self.items=[]
        for s in sc:
            m=s['mission']
            # anchor tu goal_ref hoac via_ref (near/far)
            anc=None
            for r in [m['goal_ref'],m['via_ref']]:
                if r and r.get('kind') in ('near','far'): anc=r.get('anchor')
            if anc: self.items.append((s['mission']['text'],TYPES.index(anc)))
            # them negative? Khong, chi train tren near/far (325 train). Val danh gia tren near/far.
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]
def run(epochs=8,bs=16):
    tok=AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
    dtr=ADS('train'); dva=ADS('validation')
    print(f'anchor train {len(dtr)} val {len(dva)}',flush=True)
    dev=torch.device('cuda')
    encm=AutoModel.from_pretrained('vinai/phobert-base-v2')
    head=nn.Linear(encm.config.hidden_size,10)
    m=nn.ModuleDict({'enc':encm,'head':head}).to(dev)
    # gop params
    opt=torch.optim.AdamW(list(encm.parameters())+list(head.parameters()),lr=2e-5)
    ce=nn.CrossEntropyLoss()
    def enc_batch(texts):
        return tok(texts,padding=True,truncation=True,max_length=128,return_tensors='pt')
    import random
    for ep in range(epochs):
        random.shuffle(dtr.items)
        tot=0; m.train()
        for i in range(0,len(dtr),bs):
            batch=dtr.items[i:i+bs]
            texts=[strip(t) if random.random()<0.3 else t for t,_ in batch]
            y=torch.tensor([y for _,y in batch]).to(dev)
            e=enc_batch(texts); e={k:v.to(dev) for k,v in e.items()}
            p=m['enc'](**{k:v for k,v in e.items() if k in ('input_ids','attention_mask')}).last_hidden_state[:,0]
            loss=ce(m['head'](p),y)
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
        print(f'ep{ep+1} loss={tot/max(1,len(dtr)//bs):.3f}',flush=True)
        # val
        m.eval(); hit=0
        with torch.no_grad():
            for i in range(0,len(dva),32):
                batch=dva.items[i:i+32]
                e=enc_batch([t for t,_ in batch]); e={k:v.to(dev) for k,v in e.items()}
                p=m['enc'](**{k:v for k,v in e.items() if k in ('input_ids','attention_mask')}).last_hidden_state[:,0]
                hit+=(m['head'](p).argmax(1).cpu()==torch.tensor([y for _,y in batch])).sum().item()
        print(f' anchor val={hit/max(1,len(dva)):.4f} ({hit}/{len(dva)})',flush=True)
    torch.save({'head':m['head'].state_dict()},'/mnt/hdd2/qtech/Phenika-AI/solution/nlp/anchor_head.pt')
    print('saved anchor_head (encoder dung chung phobert_mt? Tam luu head, se gop sau)')
if __name__=='__main__': run()
