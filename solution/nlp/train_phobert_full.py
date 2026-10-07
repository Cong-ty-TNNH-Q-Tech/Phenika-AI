"""Train PhoBERT full multitask - chay that tren GPU2."""
import json, pathlib, random, unicodedata
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
LABELS=['library','dorm','sports','clinic','canteen','parking','lecture','lab','office','gate']
REFS=['NONE','north','south','west','east','near','far']
def strip(s): return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
class DS(Dataset):
    def __init__(self,scenes,train=True): self.s=scenes; self.tr=train
    def __len__(self): return len(self.s)
    def __getitem__(self,i):
        s=self.s[i]; t=s['mission']['text']
        if self.tr:
            r=random.random()
            if r<0.25: t=strip(t)
            elif r<0.35: t=t.lower()
        m=s['mission']
        return (t, LABELS.index(m['goal']), LABELS.index(m['via']) if m['via'] else 10,
                int(m['urgent']), int(m['fragile']),
                REFS.index(m['goal_ref']['kind']) if m['goal_ref'] else 0)
def coll(batch, tok):
    texts=[b[0] for b in batch]
    enc=tok(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    return enc, torch.tensor([b[1] for b in batch]), torch.tensor([b[2] for b in batch]), torch.tensor([b[3] for b in batch]), torch.tensor([b[4] for b in batch]), torch.tensor([b[5] for b in batch])
class MT(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=AutoModel.from_pretrained('vinai/phobert-base-v2')
        h=self.enc.config.hidden_size
        self.g=nn.Linear(h,10); self.v=nn.Linear(h,11); self.u=nn.Linear(h,2); self.f=nn.Linear(h,2); self.r=nn.Linear(h,7)
    def forward(self, enc):
        p=self.enc(**{k:v for k,v in enc.items() if k in ('input_ids','attention_mask')}).last_hidden_state[:,0]
        return self.g(p), self.v(p), self.u(p), self.f(p), self.r(p)
def run(epochs=3, bs=16):
    tok=AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
    tr=json.loads((BASE/'train/scenes.json').read_text(encoding='utf-8'))
    va=json.loads((BASE/'validation/scenes.json').read_text(encoding='utf-8'))
    dtr=DS(tr,True); dva=DS(va,False)
    ltr=DataLoader(dtr,batch_size=bs,shuffle=True,collate_fn=lambda b:coll(b,tok))
    lva=DataLoader(dva,batch_size=32,shuffle=False,collate_fn=lambda b:coll(b,tok))
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',dev, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
    m=MT().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=2e-5)
    ce=nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); tot=0
        for enc,g,v,u,f,r in ltr:
            enc={k:v.to(dev) for k,v in enc.items()}; g,v,u,f,r=g.to(dev),v.to(dev),u.to(dev),f.to(dev),r.to(dev)
            pg,pv,pu,pf,pr=m(enc); loss=ce(pg,g)+ce(pv,v)+ce(pu,u)+ce(pf,f)+ce(pr,r)
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
        print(f'ep{ep+1}/{epochs} loss={tot/len(ltr):.3f}',flush=True)
        # val goal acc
        m.eval(); hit=0; tot2=0
        with torch.no_grad():
            for enc,g,v,u,f,r in lva:
                enc={k:v.to(dev) for k,v in enc.items()}
                pg,_,_,_,_=m(enc); hit+=(pg.argmax(1).cpu()==g).sum().item(); tot2+=len(g)
        print(f' val goal={hit/tot2:.4f}',flush=True)
    torch.save(m.state_dict(),'/mnt/hdd2/qtech/Phenika-AI/solution/nlp/phobert_mt.pt')
    print('saved solution/nlp/phobert_mt.pt')
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--batch',type=int,default=16)
    a=ap.parse_args(); run(a.epochs,a.batch)
