"""Train edge-PRESENCE classifier: crop giua 2 node ke -> co duong hay khong. ResNet18 binary."""
import json, pathlib, random, collections
import torch, torch.nn as nn, numpy as np
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TF=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def build_items(split,neg_per_scene=40,seed=0):
    rng=random.Random(seed)
    sc=json.loads((BASE/f'{split}/scenes.json').read_text())
    items=[]  # (image, x0,y0, label)
    for s in sc:
        nodes={tuple(n['rc']):n['xy'] for n in s['nodes']}
        edges={tuple(sorted([tuple(e['a']),tuple(e['b'])])) for e in s['edges']}
        negs=[]
        for (rc,xy) in nodes.items():
            for dr,dc in [(0,1),(1,0)]:
                nb=(rc[0]+dr,rc[1]+dc)
                if nb in nodes:
                    key=tuple(sorted([rc,nb])); xy2=nodes[nb]
                    mx,my=(xy[0]+xy2[0])/2,(xy[1]+xy2[1])/2
                    lab=1 if key in edges else 0
                    if lab==0: negs.append((mx,my))
                    else: items.append((s['image'],int(mx-32),int(my-32),1))
        rng.shuffle(negs)
        for mx,my in negs[:neg_per_scene]:
            items.append((s['image'],int(mx-32),int(my-32),0))
    return items
class DS(Dataset):
    def __init__(self,items,split):
        self.items=items; self.split=split
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        fn,x0,y0,y=self.items[i]
        im=Image.open(BASE/self.split/fn).convert('RGB'); w,h=im.size
        x0,y0=max(0,x0),max(0,y0)
        return TF(im.crop((x0,y0,min(w,x0+64),min(h,y0+64)))),y
def run(epochs=4):
    dev=torch.device('cuda')
    print('building items...',flush=True)
    itr=build_items('train'); iva=build_items('validation',seed=1)
    npos=sum(y for *_,y in itr); print(f'train {len(itr)} pos={npos} neg={len(itr)-npos}',flush=True)
    ltr=DataLoader(DS(itr,'train'),batch_size=128,shuffle=True,num_workers=8,pin_memory=True)
    lva=DataLoader(DS(iva,'validation'),batch_size=128,num_workers=8,pin_memory=True)
    m=models.resnet18(weights='IMAGENET1K_V1'); m.fc=nn.Linear(m.fc.in_features,2); m=m.to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3); w=torch.tensor([1.0, npos and (len(itr)-npos)/npos or 1.0]).to(dev)
    ce=nn.CrossEntropyLoss(weight=w)
    best=-1; save='/mnt/hdd2/qtech/Phenika-AI/solution/cv/edge_presence.pt'
    for ep in range(epochs):
        m.train(); tot=0
        for x,y in ltr:
            x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True)
            opt.zero_grad(); loss=ce(m(x),y); loss.backward(); opt.step(); tot+=loss.item()
        m.eval(); hit=0; tot2=0; pos_hit=0;pos_tot=0;neg_hit=0;neg_tot=0
        with torch.no_grad():
            for x,y in lva:
                x=x.to(dev); p=m(x).argmax(1).cpu()
                hit+=(p==y).sum().item(); tot2+=len(y)
                pos_hit+=((p==1)&(y==1)).sum().item(); pos_tot+=(y==1).sum().item()
                neg_hit+=((p==0)&(y==0)).sum().item(); neg_tot+=(y==0).sum().item()
        acc=hit/tot2
        print(f'ep{ep+1} loss={tot/len(ltr):.3f} val={acc:.4f} pos_recall={pos_hit/max(1,pos_tot):.3f} neg_recall={neg_hit/max(1,neg_tot):.3f}',flush=True)
        if acc>best:
            best=acc; torch.save(m.state_dict(),save); print(f'  saved best={best:.4f}',flush=True)
    print('best presence val',round(best,4),'->',save)
if __name__=='__main__': run()
