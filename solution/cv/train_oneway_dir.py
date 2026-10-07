"""Oneway 3 lop: 0=2chieu, 1=a->b, 2=b->a. Crop doc theo edge + mui ten."""
import json, pathlib
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TF=transforms.Compose([transforms.Resize((64,128)),transforms.ToTensor()])
class OWDS(Dataset):
    def __init__(self,split,n=15000):
        self.items=[]
        sc=json.loads((BASE/split/'scenes.json').read_text())
        for s in sc:
            if len(self.items)>=n: break
            xy={tuple(x['rc']):x['xy'] for x in s['nodes']}
            for e in s['edges']:
                if len(self.items)>=n: break
                a,b=tuple(e['a']),tuple(e['b'])
                # can bang: lay het oneway (5%) + sample 2chieu
                is_one=e['oneway_to'] is not None
                if not is_one and len(self.items)%3!=0: continue
                pa,pb=xy[a],xy[b]
                # crop doc: xoay? don gian crop 64x128 quanh midpoint (giu huong doc/ngang bang cach sap xep a<b?)
                mx,my=(pa[0]+pb[0])/2,(pa[1]+pb[1])/2
                if is_one:
                    t=tuple(e['oneway_to'])
                    y=1 if t==b else 2
                else: y=0
                self.items.append((s['image'],split,(int(mx-32),int(my-64)),y))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        fn,split,(x0,y0),y=self.items[i]
        im=Image.open(BASE/split/fn).convert('RGB')
        w,h=im.size; x0,y0=max(0,x0),max(0,y0)
        return TF(im.crop((x0,y0,min(w,x0+64),min(h,y0+128)))),y
def run(epochs=5,bs=128):
    dev=torch.device('cuda')
    dtr=OWDS('train'); dva=OWDS('validation',3000)
    from collections import Counter
    print(f'train {len(dtr)} {dict(Counter(y for _,_,_,y in dtr.items))}',flush=True)
    ltr=DataLoader(dtr,batch_size=bs,shuffle=True,num_workers=2); lva=DataLoader(dva,batch_size=bs,num_workers=2)
    m=models.mobilenet_v3_small(weights='IMAGENET1K_V1'); m.classifier[3]=nn.Linear(m.classifier[3].in_features,3); m=m.to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3); ce=nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); tot=0
        for x,y in ltr:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); loss=ce(m(x),y); loss.backward(); opt.step(); tot+=loss.item()
        print(f'ep{ep+1} loss={tot/len(ltr):.3f}',flush=True)
        m.eval(); hit=0; tot2=0
        with torch.no_grad():
            for x,y in lva:
                x=x.to(dev); hit+=(m(x).argmax(1).cpu()==y).sum().item(); tot2+=len(y)
        print(f' oneway3 val={hit/tot2:.4f}',flush=True)
    torch.save(m.state_dict(),'/mnt/hdd2/qtech/Phenika-AI/solution/cv/oneway_dir.pt')
    print('saved oneway_dir')
if __name__=='__main__': run()
