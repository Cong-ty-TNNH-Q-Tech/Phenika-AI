"""Siamese co hoc: edge + 4 swatches -> 4 logits. Train that."""
import json, pathlib, random
from PIL import Image
import torch, torch.nn as nn
import torch.nn.functional as NF
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TF=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
KINDS=['normal','crowded','covered','closed']
class SiamDS(Dataset):
    def __init__(self,split,n=20000):
        self.items=[]
        sc=json.loads((BASE/split/'scenes.json').read_text())
        for s in sc:
            if len(self.items)>=n: break
            im=Image.open(BASE/split/s['image']).convert('RGB')
            sw={}
            for lg in s['legend']:
                if lg['kind'] in KINDS:
                    x0,y0,x1,y1=[int(v) for v in lg['swatch']]
                    sw[lg['kind']]=im.crop((x0,y0,x1,y1))
            if len(sw)<4: continue
            xy={tuple(x['rc']):x['xy'] for x in s['nodes']}
            for e in s['edges']:
                if len(self.items)>=n: break
                a,b=xy[tuple(e['a'])],xy[tuple(e['b'])]
                mx,my=(a[0]+b[0])/2,(a[1]+b[1])/2
                crop=im.crop((max(0,int(mx-32)),max(0,int(my-32)),int(mx+32),int(my+32)))
                self.items.append((crop,[sw[k] for k in KINDS],KINDS.index(e['status'])))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        crop,sws,y=self.items[i]
        return TF(crop),torch.stack([TF(s) for s in sws]),y
class Siam(nn.Module):
    def __init__(self):
        super().__init__()
        b=models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.feat=nn.Sequential(b.features,b.avgpool,nn.Flatten())
        self.fc=nn.Linear(576*2,4)
    def forward(self,edge,sws):
        # edge (B,3,64,64), sws (B,4,3,64,64)
        fe=self.feat(edge)
        out=[]
        for k in range(4):
            fs=self.feat(sws[:,k])
            out.append(self.fc(torch.cat([fe,fs],1)).unsqueeze(1)[:, :, k] if False else None)
        # don gian: logits = -||fe-fs|| (cosine) + hoc scale? Tam: linear tren concat
        logits=torch.stack([self.fc(torch.cat([fe,self.feat(sws[:,k])],1))[:,k] if False else torch.zeros(fe.size(0),device=fe.device) for k in range(4)],1)
        return logits
# Viet gon: dung prototype matching co hoc scale/shift tren cosine
class Siam2(nn.Module):
    def __init__(self):
        super().__init__()
        b=models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.feat=nn.Sequential(b.features,b.avgpool,nn.Flatten())
        self.scale=nn.Parameter(torch.tensor(10.0)); self.bias=nn.Parameter(torch.tensor(0.0))
    def forward(self,edge,sws):
        fe=NF.normalize(self.feat(edge),dim=1)
        logits=[]
        for k in range(4):
            fs=NF.normalize(self.feat(sws[:,k]),dim=1)
            logits.append((fe*fs).sum(1)*self.scale+self.bias)
        return torch.stack(logits,1)
def run(epochs=4,bs=64,ntr=20000,nva=3000):
    dev=torch.device('cuda')
    print('data...',flush=True)
    dtr=SiamDS('train',ntr); dva=SiamDS('validation',nva)
    print(f'train {len(dtr)} val {len(dva)}',flush=True)
    ltr=DataLoader(dtr,batch_size=bs,shuffle=True,num_workers=2); lva=DataLoader(dva,batch_size=bs,num_workers=2)
    m=Siam2().to(dev); opt=torch.optim.Adam(m.parameters(),lr=1e-4); ce=nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); tot=0
        for x,s,y in ltr:
            x,s,y=x.to(dev),s.to(dev),y.to(dev)
            opt.zero_grad(); loss=ce(m(x,s),y); loss.backward(); opt.step(); tot+=loss.item()
        print(f'ep{ep+1} loss={tot/len(ltr):.3f}',flush=True)
        m.eval(); hit=0; tot2=0
        with torch.no_grad():
            for x,s,y in lva:
                x,s=x.to(dev),s.to(dev)
                hit+=(m(x,s).argmax(1).cpu()==y).sum().item(); tot2+=len(y)
        print(f' siamese val={hit/tot2:.4f}',flush=True)
    torch.save(m.state_dict(),'/mnt/hdd2/qtech/Phenika-AI/solution/cv/siamese_mlp.pt')
    print('saved siamese_mlp')
if __name__=='__main__': run()
