"""Legend swatch detector (normal/crowded/covered/closed/stairs/oneway) FasterRCNN."""
import json, pathlib
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
KINDS=['normal','crowded','covered','closed','stairs','oneway']
class LegDS(Dataset):
    def __init__(self,split,n=800,sz=800):
        self.sc=json.loads((BASE/split/'scenes.json').read_text())[:n]
        self.split=split; self.sz=sz
    def __len__(self): return len(self.sc)
    def __getitem__(self,i):
        s=self.sc[i]
        im=Image.open(BASE/self.split/s['image']).convert('RGB')
        w0,h0=im.size; imr=im.resize((self.sz,self.sz))
        sx,sy=self.sz/w0,self.sz/h0
        boxes=[]; labels=[]
        for lg in s['legend']:
            if lg['kind'] not in KINDS: continue
            x0,y0,x1,y1=lg['swatch']
            boxes.append([x0*sx,y0*sy,x1*sx,y1*sy]); labels.append(KINDS.index(lg['kind'])+1)
        return F.to_tensor(imr),{'boxes':torch.tensor(boxes,dtype=torch.float32),'labels':torch.tensor(labels,dtype=torch.int64)}
def coll(b): return tuple(zip(*b))
def run(epochs=5,n=800,bs=4):
    dev=torch.device('cuda')
    print('device',dev,flush=True)
    ltr=DataLoader(LegDS('train',n),batch_size=bs,shuffle=True,collate_fn=coll,num_workers=2)
    m=fasterrcnn_resnet50_fpn(weights='DEFAULT')
    m.roi_heads.box_predictor.cls_score=torch.nn.Linear(m.roi_heads.box_predictor.cls_score.in_features,7)
    m.roi_heads.box_predictor.bbox_pred=torch.nn.Linear(m.roi_heads.box_predictor.bbox_pred.in_features,28)
    m=m.to(dev); opt=torch.optim.SGD(m.parameters(),lr=0.005,momentum=0.9,weight_decay=5e-4)
    for ep in range(epochs):
        m.train(); tot=0
        for x,y in ltr:
            x=[t.to(dev) for t in x]; y=[{k:v.to(dev) for k,v in t.items()} for t in y]
            loss=sum(v for v in m(x,y).values())
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
        print(f'ep{ep+1}/{epochs} loss={tot/len(ltr):.3f}',flush=True)
    torch.save(m.state_dict(),'/mnt/hdd2/qtech/Phenika-AI/solution/cv/legend_frcnn.pt')
    print('saved legend_frcnn')
if __name__=='__main__': run()
