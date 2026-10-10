"""One MobileNet Faster-RCNN for nodes + road legend, replacing two ResNet detectors."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

KINDS=['normal','crowded','covered','closed','stairs','oneway']


def make_model(pretrained=False):
    model=fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT' if pretrained else None,
                                          weights_backbone=None,min_size=640,max_size=1100)
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,8)
    return model


class Scenes(Dataset):
    def __init__(self,data,split):
        self.root=data/split
        self.scenes=json.loads((self.root/'scenes.json').read_text(encoding='utf8'))

    def __len__(self):return len(self.scenes)

    def __getitem__(self,index):
        scene=self.scenes[index]
        image=Image.open(self.root/scene['image']).convert('RGB')
        w,h=image.size
        boxes=[];labels=[]
        for n in scene['nodes']:
            x,y=n['xy'];boxes.append([max(0,x-10),max(0,y-10),min(w,x+10),min(h,y+10)])
            labels.append(1)
        for legend in scene['legend']:
            if legend['kind'] not in KINDS:continue
            x0,y0,x1,y1=legend['swatch']
            boxes.append([max(0,x0),max(0,y0),min(w,x1),min(h,y1)])
            labels.append(2+KINDS.index(legend['kind']))
        return to_tensor(image),{'boxes':torch.tensor(boxes,dtype=torch.float32),
                                  'labels':torch.tensor(labels,dtype=torch.long)}


def collate(batch):return tuple(zip(*batch))


class Views:
    """Two label views over one checkpoint, caching a shared image forward."""
    def __init__(self,checkpoint,device):
        self.model=make_model()
        self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        self.model.to(device).eval()
        self.cached=None

    def nodes(self,images):
        self.cached=self.model(images)
        out=[]
        for item in self.cached:
            mask=item['labels']==1
            out.append({k:v[mask] for k,v in item.items()})
        return out

    def legend(self,images):
        predictions=self.cached if self.cached is not None else self.model(images)
        self.cached=None
        out=[]
        for item in predictions:
            mask=item['labels']>=2
            parsed={k:v[mask] for k,v in item.items()}
            parsed['labels']=parsed['labels']-1
            out.append(parsed)
        return out


class View(nn.Module):
    def __init__(self,shared,kind):
        super().__init__();self.shared=shared;self.kind=kind

    def forward(self,images):return getattr(self.shared,self.kind)(images)


def train(args):
    torch.manual_seed(2026)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',dev,flush=True)
    model=make_model(args.pretrained)
    if args.init:model.load_state_dict(torch.load(args.init,map_location='cpu',weights_only=True))
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():parameter.requires_grad=False
    model.to(dev)
    loader=DataLoader(Scenes(args.data,'train'),batch_size=1,shuffle=True,
                      num_workers=args.workers,collate_fn=collate)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                              lr=.003,momentum=.9,weight_decay=.0005)
    args.out.mkdir(parents=True,exist_ok=True)
    for ep in range(args.epochs):
        model.train();total=0.
        for step,(images,targets) in enumerate(loader):
            images=[x.to(dev) for x in images]
            targets=[{k:v.to(dev) for k,v in t.items()} for t in targets]
            loss=sum(model(images,targets).values())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite detector loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=float(loss.detach())
            if (step+1)%50==0:print(f'epoch {ep+1} {step+1}/{len(loader)} loss={total/(step+1):.4f}',flush=True)
        torch.save(model.state_dict(),args.out/f'joint_detector_epoch{ep+1}.pt')
        # This is a training checkpoint. E2E validation must select the usable epoch.
    print('finished: validate detector checkpoints end-to-end before replacing the original',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('artifacts/compact_cv'))
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--pretrained',action='store_true')
    p.add_argument('--freeze-backbone',action='store_true')
    p.add_argument('--init',type=Path)
    train(p.parse_args())
