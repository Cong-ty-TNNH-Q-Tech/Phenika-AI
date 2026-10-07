"""Robot 5 lop (none + UP/DOWN/LEFT/RIGHT) + Weather 2 lop. Train lien."""
import json, pathlib
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TF=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
TFW=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
HEADS=['UP','DOWN','LEFT','RIGHT']
class RobotDS(Dataset):
    def __init__(self,split,n_scene=800):
        self.items=[]
        sc=json.loads((BASE/split/'scenes.json').read_text())[:n_scene]
        for s in sc:
            xy={tuple(x['rc']):x['xy'] for x in s['nodes']}
            rrc=tuple(s['robot']['rc']); rh=s['robot']['heading']
            im=Image.open(BASE/split/s['image']).convert('RGB')
            for rc,center in xy.items():
                y=HEADS.index(rh)+1 if rc==rrc else 0
                self.items.append((s['image'],split,(int(center[0]-32),int(center[1]-32)),y))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        fn,split,(x0,y0),y=self.items[i]
        im=Image.open(BASE/split/fn).convert('RGB')
        w,h=im.size; x0,y0=max(0,x0),max(0,y0)
        return TF(im.crop((x0,y0,min(w,x0+64),min(h,y0+64)))),y
class WeatherDS(Dataset):
    def __init__(self,split):
        self.sc=json.loads((BASE/split/'scenes.json').read_text())
        self.split=split
    def __len__(self): return len(self.sc)
    def __getitem__(self,i):
        s=self.sc[i]
        im=Image.open(BASE/self.split/s['image']).convert('RGB')
        x0,y0,x1,y1=[int(v) for v in s['weather_box']]
        cr=im.crop((max(0,x0),max(0,y0),x1,y1))
        return TFW(cr), (1 if s['weather']=='rain' else 0)
def train_robot():
    dev=torch.device('cuda')
    dtr=RobotDS('train'); dva=RobotDS('validation',150)
    print(f'robot train {len(dtr)}, val {len(dva)}')
    ltr=DataLoader(dtr,batch_size=128,shuffle=True,num_workers=2); lva=DataLoader(dva,batch_size=128,num_workers=2)
    m=models.mobilenet_v3_small(weights='IMAGENET1K_V1'); m.classifier[3]=nn.Linear(m.classifier[3].in_features,5); m=m.to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3); ce=nn.CrossEntropyLoss()
    for ep in range(4):
        m.train(); tot=0
        for x,y in ltr:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); loss=ce(m(x),y); loss.backward(); opt.step(); tot+=loss.item()
        print(f'robot ep{ep+1} loss={tot/len(ltr):.3f}',flush=True)
        m.eval(); hit=0; tot2=0
        with torch.no_grad():
            for x,y in lva:
                x=x.to(dev); hit+=(m(x).argmax(1).cpu()==y).sum().item(); tot2+=len(y)
        print(f' robot val5={hit/tot2:.4f}',flush=True)
    torch.save(m.state_dict(),'/mnt/hdd2/qtech/Phenika-AI/solution/cv/robot_mobilenet.pt')
    print('saved robot')
def train_weather():
    dev=torch.device('cuda')
    dtr=WeatherDS('train'); dva=WeatherDS('validation')
    print(f'weather train {len(dtr)}, val {len(dva)}')
    ltr=DataLoader(dtr,batch_size=64,shuffle=True,num_workers=2); lva=DataLoader(dva,batch_size=64,num_workers=2)
    m=models.mobilenet_v3_small(weights='IMAGENET1K_V1'); m.classifier[3]=nn.Linear(m.classifier[3].in_features,2); m=m.to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3); ce=nn.CrossEntropyLoss()
    for ep in range(5):
        m.train(); tot=0
        for x,y in ltr:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); loss=ce(m(x),y); loss.backward(); opt.step(); tot+=loss.item()
        print(f'wx ep{ep+1} loss={tot/len(ltr):.3f}',flush=True)
        m.eval(); hit=0; tot2=0
        with torch.no_grad():
            for x,y in lva:
                x=x.to(dev); hit+=(m(x).argmax(1).cpu()==y).sum().item(); tot2+=len(y)
        print(f' weather val={hit/tot2:.4f}',flush=True)
    torch.save(m.state_dict(),'/mnt/hdd2/qtech/Phenika-AI/solution/cv/weather_mobilenet.pt')
    print('saved weather')
if __name__=='__main__':
    train_robot(); train_weather()
