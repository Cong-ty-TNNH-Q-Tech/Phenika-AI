"""Train PhoBERT v3 multitask: goal/via/urg/frag/gref_kind/gref_anchor/vref_kind/vref_anchor."""
import json, pathlib, random, unicodedata
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/v3data/delivery_public')
LABELS=['library','dorm','sports','clinic','canteen','parking','lecture','lab','office','gate']
REFS=['NONE','north','south','west','east','near','far','north_most','south_most','west_most','east_most','anchor_near']
VREFS=['NONE','north','south','west','east','near','far']
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
            if random.random()<0.10 and len(t)>10:
                j=random.randrange(len(t)); t=t[:j]+t[j+1:]
        m=s['mission']
        gr=m['goal_ref']; vr=m['via_ref']
        return (t, LABELS.index(m['goal']), LABELS.index(m['via']) if m['via'] else 10,
                int(m['urgent']), int(m['fragile']),
                REFS.index(gr['kind']) if gr else 0,
                LABELS.index(gr['anchor']) if (gr and gr.get('anchor')) else 10,
                VREFS.index(vr['kind']) if vr else 0,
                LABELS.index(vr['anchor']) if (vr and vr.get('anchor')) else 10)
def coll(batch, tok):
    enc=tok([b[0] for b in batch], padding=True, truncation=True, max_length=160, return_tensors='pt')
    return enc, *[torch.tensor([b[j] for b in batch]) for j in range(1,9)]
class MT(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=AutoModel.from_pretrained('vinai/phobert-base-v2')
        h=self.enc.config.hidden_size
        self.g=nn.Linear(h,10); self.v=nn.Linear(h,11); self.u=nn.Linear(h,2); self.f=nn.Linear(h,2)
        self.r=nn.Linear(h,12); self.ra=nn.Linear(h,11); self.vr=nn.Linear(h,7); self.va=nn.Linear(h,11)
    def forward(self, enc):
        p=self.enc(**{k:v for k,v in enc.items() if k in ('input_ids','attention_mask')}).last_hidden_state[:,0]
        return self.g(p), self.v(p), self.u(p), self.f(p), self.r(p), self.ra(p), self.vr(p), self.va(p)
def run(epochs=10, bs=4, accum=4, patience=3, out='/mnt/hdd2/qtech/Phenika-AI/solution/nlp/phobert_v3.pt'):
    import random as _r; _r.seed(42); torch.manual_seed(42)
    tok=AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
    tr=json.loads((BASE/'train/scenes.json').read_text(encoding='utf-8'))
    va=json.loads((BASE/'validation/scenes.json').read_text(encoding='utf-8'))
    ltr=DataLoader(DS(tr,True),batch_size=bs,shuffle=True,collate_fn=lambda b:coll(b,tok))
    lva=DataLoader(DS(va,False),batch_size=32,shuffle=False,collate_fn=lambda b:coll(b,tok))
    dev=torch.device('cuda'); m=MT().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=2e-5); ce=nn.CrossEntropyLoss(); ce_none=nn.CrossEntropyLoss(reduction='none')
    best=-1; bad=0
    for ep in range(epochs):
        m.train(); tot=0; opt.zero_grad()
        for step,(enc,g,v,u,f,r,ra,vr,va) in enumerate(ltr):
            enc={k:vv.to(dev) for k,vv in enc.items()}
            g,v,u,f,r,ra,vr,va=g.to(dev),v.to(dev),u.to(dev),f.to(dev),r.to(dev),ra.to(dev),vr.to(dev),va.to(dev)
            pg,pv,pu,pf,pr,pra,pvr,pva=m(enc)
            # mask goal loss for map-based scenes (goal type not in text)
            MAPK={7,8,9,10,11}  # *_most + anchor_near indices in REFS
            is_map=torch.tensor([1 if int(x) in MAPK else 0 for x in r],device=dev,dtype=torch.float32)
            loss_pg = (ce_none(pg,g)*(1-is_map)).sum()/max(1.0,(1-is_map).sum())
            loss=(3.0*loss_pg+2.0*ce(pv,v)+ce(pu,u)+ce(pf,f)+ce(pr,r)+0.5*ce(pra,ra)+ce(pvr,vr)+0.5*ce(pva,va))/accum
            loss.backward(); tot+=loss.item()*accum
            if (step+1)%accum==0: opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        m.eval(); hg=hv=hr=hra=hvr=0; n=0
        with torch.no_grad():
            for enc,g,v,u,f,r,ra,vr,va in lva:
                enc={k:vv.to(dev) for k,vv in enc.items()}
                pg,pv,pu,pf,pr,pra,pvr,pva=m(enc)
                hg+=(pg.argmax(1).cpu()==g).sum().item(); hv+=(pv.argmax(1).cpu()==v).sum().item()
                hr+=(pr.argmax(1).cpu()==r).sum().item(); hra+=(pra.argmax(1).cpu()==ra).sum().item()
                hvr+=(pvr.argmax(1).cpu()==vr).sum().item(); n+=len(g)
        metric=(hg+hv)/2/n
        print(f'ep{ep+1} loss={tot/len(ltr):.3f} goal={hg/n:.4f} via={hv/n:.4f} gref={hr/n:.4f} ganchor={hra/n:.4f} vref={hvr/n:.4f}',flush=True)
        if metric>best: best=metric; bad=0; torch.save(m.state_dict(),out); print('  saved',flush=True)
        else:
            bad+=1
            if bad>=patience: print('early stop',flush=True); break
    print('best',best,'->',out)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--batch',type=int,default=4); ap.add_argument('--patience',type=int,default=3); ap.add_argument('--out',default='/mnt/hdd2/qtech/Phenika-AI/solution/nlp/phobert_v3.pt')
    a=ap.parse_args(); run(a.epochs,a.batch,4,a.patience,a.out)
