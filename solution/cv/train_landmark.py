"""Landmark 11 lop (10 + none) - crop tai node. Train that GPU1."""
import json, pathlib, collections
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TYPES=['library','dorm','sports','clinic','canteen','parking','lecture','lab','office','gate']
TF=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
class LMDS(Dataset):
    def __init__(self,split,n_scene=800):
        self.items=[]
        sc=json.loads((BASE/split/'scenes.json').read_text())[:n_scene]
        for s in sc:
            xy={tuple(x['rc']):x['xy'] for x in s['nodes']}
            lm={tuple(x['rc']):x['type'] for x in s['landmarks']}
            im=Image.open(BASE/split/s['image']).convert('RGB')
            for rc,center in xy.items():
                t=lm.get(rc,'NONE')
                x0,y0=int(center[0]-32),int(center[1]-32)
                self.items.append((s['image'],split,(x0,y0),t))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        fn,split,(x0,y0),t=self.items[i]
        im=Image.open(BASE/split/fn).convert('RGB')
        w,h=im.size; x0,y0=max(0,x0),max(0,y0)
        cr=im.crop((x0,y0,min(w,x0+64),min(h,y0+64)))
        y=TYPES.index(t) if t!='NONE' else 10
        return TF(cr),y
def run(epochs=4,bs=128):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',dev)
    dtr=LMDS('train'); dva=LMDS('validation',150)
    print(f'train {len(dtr)} node-crops, val {len(dva)}')
    ltr=DataLoader(dtr,batch_size=bs,shuffle=True,num_workers=2); lva=DataLoader(dva,batch_size=bs,num_workers=2)
    m=models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    m.classifier[3]=nn.Linear(m.classifier[3].in_features,11)
    m=m.to(dev); opt=torch.optim.Adam(m.parameters(),lr=1e-3); ce=nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); tot=0
        for x,y in ltr:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); loss=ce(m(x),y); loss.backward(); opt.step(); tot+=loss.item()
        print(f'ep{ep+1} loss={tot/len(ltr):.3f}',flush=True)
        m.eval(); hit=0; tot2=0
        with torch.no_grad():
            for x,y in lva:
                x=x.to(dev); hit+=(m(x).argmax(1).cpu()==y).sum().item(); tot2+=len(y)
        print(f' val landmark11={hit/tot2:.4f}',flush=True)
    torch.save(m.state_dict(),'/mnt/hdd2/qtech/Phenika-AI/solution/cv/landmark_mobilenet.pt')
    print('saved solution/cv/landmark_mobilenet.pt')
if __name__=='__main__': run()
