"""Node detector FasterRCNN - train + early stopping (best F1 val) tren GPU."""
import json, pathlib
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
        w0,h0=im.size; im=im.resize((self.sz,self.sz)); sx,sy=self.sz/w0,self.sz/h0
        boxes=[[nd['xy'][0]*sx-12,nd['xy'][1]*sy-12,nd['xy'][0]*sx+12,nd['xy'][1]*sy+12] for nd in s['nodes']]
        return F.to_tensor(im), {'boxes':torch.tensor(boxes,dtype=torch.float32),'labels':torch.ones(len(boxes),dtype=torch.int64)}
def coll(b): return tuple(zip(*b))
def iou(a,b):
    xa=max(a[0],b[0]); ya=max(a[1],b[1]); xb=min(a[2],b[2]); yb=min(a[3],b[3])
    inter=max(0,xb-xa)*max(0,yb-ya)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0
def evaluate(m,dva,dev,maxn=80):
    m.eval(); tp=fp=fn=0
    with torch.no_grad():
        for i in range(min(maxn,len(dva.sc))):
            x,y=dva[i]; out=m([x.to(dev)])[0]
            pred=[b.tolist() for b,s,l in zip(out['boxes'].cpu(),out['scores'].cpu(),out['labels'].cpu()) if l==1 and s>0.5]
            gt=y['boxes'].tolist(); used=set()
            for pb in pred:
                bi=-1; bv=0.3
                for gi,gb in enumerate(gt):
                    if gi in used: continue
                    v=iou(pb,gb)
                    if v>bv: bv=v; bi=gi
                if bi>=0: used.add(bi); tp+=1
                else: fp+=1
            fn+=len(gt)-len(used)
    prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
    return 2*prec*rec/max(1e-9,prec+rec), prec, rec
def run(epochs=10,n=800,bs=4,patience=3):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',dev,flush=True)
    dtr=NodeDS('train',n); ltr=DataLoader(dtr,batch_size=bs,shuffle=True,collate_fn=coll,num_workers=2)
    dva=NodeDS('validation',120)
    m=fasterrcnn_resnet50_fpn(weights='DEFAULT')
    m.roi_heads.box_predictor.cls_score=torch.nn.Linear(m.roi_heads.box_predictor.cls_score.in_features,2)
    m.roi_heads.box_predictor.bbox_pred=torch.nn.Linear(m.roi_heads.box_predictor.bbox_pred.in_features,8)
    m=m.to(dev); opt=torch.optim.SGD(m.parameters(),lr=0.005,momentum=0.9,weight_decay=5e-4)
    best=-1; bad=0; save='/mnt/hdd2/qtech/Phenika-AI/solution/cv/nodes_frcnn.pt'
    for ep in range(epochs):
        m.train(); tot=0
        for x,y in ltr:
            x=[t.to(dev) for t in x]; y=[{k:v.to(dev) for k,v in t.items()} for t in y]
            loss=sum(v for v in m(x,y).values()); opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
        f1,prec,rec=evaluate(m,dva,dev)
        print(f'ep{ep+1}/{epochs} loss={tot/len(ltr):.3f} val_F1={f1:.4f} P={prec:.3f} R={rec:.3f}',flush=True)
        if f1>best:
            best=f1; bad=0; torch.save(m.state_dict(),save); print(f'  saved best F1={best:.4f}',flush=True)
        else:
            bad+=1
            if bad>=patience: print('early stop',flush=True); break
    print('best val_F1',round(best,4),'->',save)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--n',type=int,default=800); ap.add_argument('--patience',type=int,default=3)
    a=ap.parse_args(); run(a.epochs,a.n,4,a.patience)
