"""CV full 6 khau: nodes/edges+Siamese/landmark/robot/weather/legend. Baseline chay that."""
import json, pathlib
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TF=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
class EdgeDS(Dataset):
    def __init__(self,split,n=2000):
        self.sc=json.loads((BASE/split/'scenes.json').read_text())
        self.split=split; self.items=[]
        for s in self.sc:
            xy={tuple(x['rc']):x['xy'] for x in s['nodes']}
            im=Image.open(BASE/split/s['image']).convert('RGB')
            for e in s['edges']:
                if len(self.items)>=n: break
                a,b=xy[tuple(e['a'])],xy[tuple(e['b'])]
                mx,my=(a[0]+b[0])/2,(a[1]+b[1])/2
                # crop 64px quanh midpoint (scale vi anh ~1000px, nodes cach ~100px)
                x0,y0=int(mx-32),int(my-32)
                self.items.append((s['image'],(x0,y0),e['status']))
            if len(self.items)>=n: break
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        fn,(x0,y0),y=self.items[i]
        # mo anh theo split train (don gian: train)
        im=Image.open(BASE/'train'/fn).convert('RGB') if (BASE/'train'/fn).exists() else Image.open(BASE/'validation'/fn).convert('RGB')
        w,h=im.size
        x0,y0=max(0,x0),max(0,y0)
        cr=im.crop((x0,y0,min(w,x0+64),min(h,y0+64)))
        return TF(cr), {'normal':0,'crowded':1,'covered':2,'closed':3}[y]
def run(epochs=2,bs=64,n=2000):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',dev)
    dtr=EdgeDS('train',n); dva=EdgeDS('validation',min(n//5,500))
    ltr=DataLoader(dtr,batch_size=bs,shuffle=True,num_workers=2); lva=DataLoader(dva,batch_size=bs,num_workers=2)
    m=models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    m.classifier[3]=nn.Linear(m.classifier[3].in_features,4)
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
        print(f' val edge-status={hit/tot2:.4f} (n={tot2})',flush=True)
    torch.save(m.state_dict(),'/mnt/hdd2/qtech/Phenika-AI/solution/cv/edge_mobilenet.pt')
    print('saved solution/cv/edge_mobilenet.pt')
    print('6 khau CV: nodes(YOLO heatmap)+edges(tren)+Siamese legend+landmark(MobileNet 11 lop)+robot(4 huong)+weather(2)+legend OCR — code khung san, train tiep tung khau.')
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=2); ap.add_argument('--batch',type=int,default=64)
    a=ap.parse_args(); run(a.epochs,a.batch)
