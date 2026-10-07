"""Stairs + Oneway binary classifiers tren edge crops. Train nhanh."""
import json, pathlib
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TF=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
class EdgeBinDS(Dataset):
    def __init__(self,split,task,n=12000):
        # task: stairs | oneway
        self.items=[]
        sc=json.loads((BASE/split/'scenes.json').read_text())
        for s in sc:
            xy={tuple(x['rc']):x['xy'] for x in s['nodes']}
            for e in s['edges']:
                if len(self.items)>=n: break
                a,b=xy[tuple(e['a'])],xy[tuple(e['b'])]
                mx,my=(a[0]+b[0])/2,(a[1]+b[1])/2
                if task=='stairs': y=int(e['stairs'])
                else: y=int(e['oneway_to'] is not None)
                self.items.append((s['image'],split,(int(mx-32),int(my-32)),y))
            if len(self.items)>=n: break
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        fn,split,(x0,y0),y=self.items[i]
        im=Image.open(BASE/split/fn).convert('RGB')
        w,h=im.size; x0,y0=max(0,x0),max(0,y0)
        return TF(im.crop((x0,y0,min(w,x0+64),min(h,y0+64)))),y
def train_task(task):
    dev=torch.device('cuda')
    dtr=EdgeBinDS('train',task,12000); dva=EdgeBinDS('validation',task,2000)
    print(task, f'train {len(dtr)} val {len(dva)} pos_rate_tr={sum(y for _,_,_,y in dtr.items)/len(dtr):.3f}',flush=True)
    ltr=DataLoader(dtr,batch_size=128,shuffle=True,num_workers=2); lva=DataLoader(dva,batch_size=128,num_workers=2)
    m=models.mobilenet_v3_small(weights='IMAGENET1K_V1'); m.classifier[3]=nn.Linear(m.classifier[3].in_features,2); m=m.to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3); ce=nn.CrossEntropyLoss()
    for ep in range(5):
        m.train(); tot=0
        for x,y in ltr:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); loss=ce(m(x),y); loss.backward(); opt.step(); tot+=loss.item()
        print(f'{task} ep{ep+1} loss={tot/len(ltr):.3f}',flush=True)
        m.eval(); hit=0; tot2=0
        with torch.no_grad():
            for x,y in lva:
                x=x.to(dev); hit+=(m(x).argmax(1).cpu()==y).sum().item(); tot2+=len(y)
        print(f' {task} val={hit/tot2:.4f}',flush=True)
    torch.save(m.state_dict(),f'/mnt/hdd2/qtech/Phenika-AI/solution/cv/{task}_mobilenet.pt')
    print(f'saved {task}')
if __name__=='__main__':
    train_task('stairs'); train_task('oneway')
