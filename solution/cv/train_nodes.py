"""Node detector FasterRCNN - train that tren GPU1."""
import json, pathlib, random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
class NodeDS(Dataset):
    def __init__(self,split,n=500,sz=800):
        self.sc=json.loads((BASE/split/'scenes.json').read_text())[:n]
        self.split=split; self.sz=sz
    def __len__(self): return len(self.sc)
    def __getitem__(self,i):
        s=self.sc[i]
        im=Image.open(BASE/self.split/s['image']).convert('RGB')
        w0,h0=im.size
        im=im.resize((self.sz,self.sz))
        sx,sy=self.sz/w0,self.sz/h0
        boxes=[]
        for nd in s['nodes']:
            x,y=nd['xy'][0]*sx,nd['xy'][1]*sy
            boxes.append([x-12,y-12,x+12,y+12])
        boxes=torch.tensor(boxes,dtype=torch.float32)
        return F.to_tensor(im), {'boxes':boxes,'labels':torch.ones(len(boxes),dtype=torch.int64)}
def coll(b): return tuple(zip(*b))
def run(epochs=3,n=500,bs=4):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',dev)
    dtr=NodeDS('train',n); ltr=DataLoader(dtr,batch_size=bs,shuffle=True,collate_fn=coll,num_workers=2)
    m=fasterrcnn_resnet50_fpn(weights='DEFAULT')
    m.roi_heads.box_predictor.cls_score=torch.nn.Linear(m.roi_heads.box_predictor.cls_score.in_features,2)
    m.roi_heads.box_predictor.bbox_pred=torch.nn.Linear(m.roi_heads.box_predictor.bbox_pred.in_features,8)
    m=m.to(dev); opt=torch.optim.SGD(m.parameters(),lr=0.005,momentum=0.9,weight_decay=5e-4)
    for ep in range(epochs):
        m.train(); tot=0
        for x,y in ltr:
            x=[t.to(dev) for t in x]; y=[{k:v.to(dev) for k,v in t.items()} for t in y]
            loss=sum(v for v in m(x,y).values())
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
        print(f'ep{ep+1}/{epochs} loss={tot/len(ltr):.3f}',flush=True)
    torch.save(m.state_dict(),'/mnt/hdd2/qtech/Phenika-AI/solution/cv/nodes_frcnn.pt')
    print('saved solution/cv/nodes_frcnn.pt')
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--n',type=int,default=500)
    a=ap.parse_args(); run(a.epochs,a.n)
