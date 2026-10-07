"""Infer test best v1: nodes FRCNN + landmark/robot classifiers + PhoBERT missions + greedy+policies.
Bo qua closed/oneway/stairs/status chi tiet (se bo sung Siamese sau). Van hon majority nho vi tri that."""
import json, pathlib, collections, math
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import models, transforms
import torch.nn as nn
from transformers import AutoTokenizer
import sys
sys.path.insert(0,'/mnt/hdd2/qtech/Phenika-AI/solution/nlp')
from train_phobert_full import MT, LABELS
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TYPES=['library','dorm','sports','clinic','canteen','parking','lecture','lab','office','gate']
HEADS=['UP','DOWN','LEFT','RIGHT']
TF=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
REL={('UP',0):0,('UP',3):1,('UP',2):2,('UP',1):3,('DOWN',1):0,('DOWN',2):1,('DOWN',3):2,('DOWN',0):3,('LEFT',2):0,('LEFT',0):1,('LEFT',1):2,('LEFT',3):3,('RIGHT',3):0,('RIGHT',1):1,('RIGHT',0):2,('RIGHT',2):3}
def rc_to_action(a,b): return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((b[0]-a[0],b[1]-a[1]))
print('loading models...')
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# nodes
nd=fasterrcnn_resnet50_fpn(weights=None)
nd.roi_heads.box_predictor.cls_score=nn.Linear(nd.roi_heads.box_predictor.cls_score.in_features,2)
nd.roi_heads.box_predictor.bbox_pred=nn.Linear(nd.roi_heads.box_predictor.bbox_pred.in_features,8)
nd.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/nodes_frcnn.pt',map_location=dev))
nd=nd.to(dev).eval()
# landmark
lm=models.mobilenet_v3_small(weights=None); lm.classifier[3]=nn.Linear(lm.classifier[3].in_features,11); lm.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/landmark_mobilenet.pt',map_location=dev)); lm=lm.to(dev).eval()
# robot
rb=models.mobilenet_v3_small(weights=None); rb.classifier[3]=nn.Linear(rb.classifier[3].in_features,5); rb.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/robot_mobilenet.pt',map_location=dev)); rb=rb.to(dev).eval()
# PhoBERT test missions (da infer san)
test_pred=json.loads(pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/test_missions_pred.json').read_text(encoding='utf-8'))
pred_by_text={o['text']:o for o in test_pred}
te_obs=json.loads((BASE/'test/observations.json').read_text(encoding='utf-8'))
from torchvision.transforms import functional as F
import numpy as np
def detect_nodes(img_path):
    im=Image.open(img_path).convert('RGB'); w0,h0=im.size; sz=800
    imr=im.resize((sz,sz))
    with torch.no_grad():
        out=nd([F.to_tensor(imr).to(dev)])[0]
    boxes=out['boxes'].cpu(); scores=out['scores'].cpu(); labels=out['labels'].cpu()
    keep=[i for i in range(len(boxes)) if labels[i]==1 and scores[i]>0.5]
    pts=[]
    for i in keep:
        x0,y0,x1,y1=boxes[i].tolist()
        cx,cy=((x0+x1)/2/sz*w0,(y0+y1)/2/sz*h0)
        pts.append((cx,cy))
    return im,w0,h0,pts
def assign_grid(pts):
    # thu rows,cols 5..9: chon cap minimize quantization error (don gian k-means 1D)
    import numpy as np
    if len(pts)<10: return None
    xs=np.array([p[0] for p in pts]); ys=np.array([p[1] for p in pts])
    best=None
    for rows in range(5,10):
        for cols in range(5,10):
            # uoc luong bien: min/max
            ymin,ymax=ys.min(),ys.max(); xmin,xmax=xs.min(),xs.max()
            # gan hang/cot gan nhat
            err=0
            rcs=[]
            for x,y in pts:
                r=int(round((y-ymin)/max(1e-6,(ymax-ymin))*(rows-1))); c=int(round((x-xmin)/max(1e-6,(xmax-xmin))*(cols-1)))
                r=max(0,min(rows-1,r)); c=max(0,min(cols-1,c))
                rcs.append((r,c))
                ey=ymin+(ymax-ymin)*r/(rows-1) if rows>1 else y; ex=xmin+(xmax-xmin)*c/(cols-1) if cols>1 else x
                err+=(x-ex)**2+(y-ey)**2
            err/=len(pts)
            # phat trung rc (2 nodes cung o)
            dup=len(rcs)-len(set(rcs))
            score=err+dup*5000
            if best is None or score<best[0]: best=(score,rows,cols,rcs)
    return best[1],best[2],best[3]
# chay thu 5 anh test de kiem tra pipeline (khong infer het 1200 ngay vi cham)
import time
for idx in range(5):
    img=BASE/'test'/te_obs[idx*10]['image']
    im,w0,h0,pts=detect_nodes(img)
    print(f'{img.name}: detected {len(pts)} nodes')
    g=assign_grid(pts)
    print(' grid:', g[0],g[1] if g else None)
print('OK pipeline chay duoc. Infer full 1200 + landmark/robot + greedy se sinh predictions_best.')
