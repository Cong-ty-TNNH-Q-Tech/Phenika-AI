"""Weather image-level 2 lop (de infer test khong can box). Train nhanh."""
import json, pathlib
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TF=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
class WDS(Dataset):
    def __init__(self,split):
        self.sc=json.loads((BASE/split/'scenes.json').read_text())
        self.split=split
    def __len__(self): return len(self.sc)
    def __getitem__(self,i):
        s=self.sc[i]
        im=Image.open(BASE/self.split/s['image']).convert('RGB')
        return TF(im), (1 if s['weather']=='rain' else 0)
def run():
    dev=torch.device('cuda')
    ltr=DataLoader(WDS('train'),batch_size=32,shuffle=True,num_workers=2)
    lva=DataLoader(WDS('validation'),batch_size=32,num_workers=2)
    m=models.mobilenet_v3_small(weights='IMAGENET1K_V1'); m.classifier[3]=nn.Linear(m.classifier[3].in_features,2); m=m.to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3); ce=nn.CrossEntropyLoss()
    for ep in range(5):
        m.train(); tot=0
        for x,y in ltr:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); loss=ce(m(x),y); loss.backward(); opt.step(); tot+=loss.item()
        print(f'ep{ep+1} loss={tot/len(ltr):.3f}',flush=True)
        m.eval(); hit=0; tot2=0
        import torch as T
        with T.no_grad():
            for x,y in lva:
                x=x.to(dev); hit+=(m(x).argmax(1).cpu()==y).sum().item(); tot2+=len(y)
        print(f' weather-img val={hit/tot2:.4f}',flush=True)
    torch.save(m.state_dict(),'/mnt/hdd2/qtech/Phenika-AI/solution/cv/weather_img.pt')
    print('saved weather_img')
if __name__=='__main__': run()
