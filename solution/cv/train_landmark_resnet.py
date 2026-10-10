"""Landmark 11 lop ResNet18 + augment, full train. Crop tai node that."""
import json, pathlib
import torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TYPES=['library','dorm','sports','clinic','canteen','parking','lecture','lab','office','gate']
tr_aug=transforms.Compose([transforms.Resize((72,72)),transforms.RandomCrop(64),
    transforms.ColorJitter(0.3,0.3,0.3,0.05),transforms.RandomRotation(8),
    transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ev_tf=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
class DS(Dataset):
    def __init__(self,split,train,n_scene=None):
        self.items=[]
        sc=json.loads((BASE/f'{split}/scenes.json').read_text())
        if n_scene: sc=sc[:n_scene]
        for s in sc:
            lm={tuple(x['rc']):x['type'] for x in s['landmarks']}
            for n in s['nodes']:
                t=lm.get(tuple(n['rc']),'NONE')
                self.items.append((s['image'],split,tuple(n['xy']), TYPES.index(t) if t!='NONE' else 10))
        self.train=train
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        fn,split,(cx,cy),y=self.items[i]
        im=Image.open(BASE/split/fn).convert('RGB'); w,h=im.size
        x0,y0=max(0,int(cx-36)),max(0,int(cy-36))
        cr=im.crop((x0,y0,min(w,x0+72),min(h,y0+72)))
        return (tr_aug if self.train else ev_tf)(cr),y
def run(epochs=6):
    dev=torch.device('cuda')
    dtr=DS('train',True); dva=DS('validation',False)
    print(f'train {len(dtr)} val {len(dva)}',flush=True)
    ltr=DataLoader(dtr,batch_size=128,shuffle=True,num_workers=8,pin_memory=True)
    lva=DataLoader(dva,batch_size=128,num_workers=8,pin_memory=True)
    m=models.resnet18(weights='IMAGENET1K_V1'); m.fc=nn.Linear(m.fc.in_features,11); m=m.to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=5e-4); ce=nn.CrossEntropyLoss()
    best=-1; bad=0; save='/mnt/hdd2/qtech/Phenika-AI/solution/cv/landmark_resnet18.pt'
    for ep in range(epochs):
        m.train(); tot=0
        for x,y in ltr:
            x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True)
            opt.zero_grad(); loss=ce(m(x),y); loss.backward(); opt.step(); tot+=loss.item()
        m.eval(); hit=tot2=0
        with torch.no_grad():
            for x,y in lva:
                x=x.to(dev); hit+=(m(x).argmax(1).cpu()==y).sum().item(); tot2+=len(y)
        acc=hit/tot2
        print(f'ep{ep+1} loss={tot/len(ltr):.3f} val={acc:.4f}',flush=True)
        if acc>best:
            best=acc; bad=0; torch.save(m.state_dict(),save); print(f'  saved best={best:.4f}',flush=True)
        else:
            bad+=1
            if bad>=3: print('early stop',flush=True); break
    print('best landmark val',round(best,4),'->',save)
if __name__=='__main__': run()
