"""Thu ResNet18 cho edge-status 4 lop (normal/crowded/covered/closed). So voi MobileNet baseline."""
import json, pathlib
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TF=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
class EdgeDS(Dataset):
    def __init__(self,split,n=20000):
        self.items=[]
        sc=json.loads((BASE/split/'scenes.json').read_text())
        for s in sc:
            xy={tuple(x['rc']):x['xy'] for x in s['nodes']}
            for e in s['edges']:
                if len(self.items)>=n: break
                a,b=xy[tuple(e['a'])],xy[tuple(e['b'])]
                mx,my=(a[0]+b[0])/2,(a[1]+b[1])/2
                self.items.append((s['image'],split,(int(mx-32),int(my-32)),e['status']))
            if len(self.items)>=n: break
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        fn,split,(x0,y0),y=self.items[i]
        im=Image.open(BASE/split/fn).convert('RGB')
        w,h=im.size; x0,y0=max(0,x0),max(0,y0)
        cr=im.crop((x0,y0,min(w,x0+64),min(h,y0+64)))
        return TF(cr), {'normal':0,'crowded':1,'covered':2,'closed':3}[y]
def run(epochs=4,bs=128,ntr=20000,nva=4000):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',dev,flush=True)
    dtr=EdgeDS('train',ntr); dva=EdgeDS('validation',nva)
    print(f'train {len(dtr)} val {len(dva)}',flush=True)
    ltr=DataLoader(dtr,batch_size=bs,shuffle=True,num_workers=4,pin_memory=True)
    lva=DataLoader(dva,batch_size=bs,num_workers=4,pin_memory=True)
    m=models.resnet18(weights='IMAGENET1K_V1')
    m.fc=nn.Linear(m.fc.in_features,4)
    n_params=sum(p.numel() for p in m.parameters())
    print(f'ResNet18 params={n_params/1e6:.1f}M',flush=True)
    m=m.to(dev); opt=torch.optim.Adam(m.parameters(),lr=1e-3); ce=nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); tot=0
        for x,y in ltr:
            x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True)
            opt.zero_grad(); loss=ce(m(x),y); loss.backward(); opt.step(); tot+=loss.item()
        print(f'ep{ep+1} loss={tot/len(ltr):.3f}',flush=True)
        m.eval(); hit=0; tot2=0
        with torch.no_grad():
            for x,y in lva:
                x=x.to(dev); hit+=(m(x).argmax(1).cpu()==y).sum().item(); tot2+=len(y)
        print(f' val edge-status ResNet18={hit/tot2:.4f} (n={tot2})',flush=True)
    torch.save(m.state_dict(),'/mnt/hdd2/qtech/Phenika-AI/solution/cv/edge_resnet18.pt')
    print('saved solution/cv/edge_resnet18.pt',flush=True)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=4)
    ap.add_argument('--ntr',type=int,default=20000); ap.add_argument('--nva',type=int,default=4000)
    a=ap.parse_args(); run(a.epochs,128,a.ntr,a.nva)
