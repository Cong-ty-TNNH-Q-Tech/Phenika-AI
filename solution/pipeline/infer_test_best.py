"""Infer test best full 1200: nodes + landmark/robot + PhoBERT + greedy policies."""
import json, pathlib, collections
import torch
from PIL import Image
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import models, transforms
from torchvision.transforms import functional as F
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TYPES=['library','dorm','sports','clinic','canteen','parking','lecture','lab','office','gate']
HEADS=['UP','DOWN','LEFT','RIGHT']
TF=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
REL={('UP',0):0,('UP',3):1,('UP',2):2,('UP',1):3,('DOWN',1):0,('DOWN',2):1,('DOWN',3):2,('DOWN',0):3,('LEFT',2):0,('LEFT',0):1,('LEFT',1):2,('LEFT',3):3,('RIGHT',3):0,('RIGHT',1):1,('RIGHT',0):2,('RIGHT',2):3}
HEAD2DIR={'UP':0,'DOWN':1,'LEFT':2,'RIGHT':3}
def rc_to_action(a,b): return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((b[0]-a[0],b[1]-a[1]))
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('load...',dev,flush=True)
nd=fasterrcnn_resnet50_fpn(weights=None)
nd.roi_heads.box_predictor.cls_score=nn.Linear(nd.roi_heads.box_predictor.cls_score.in_features,2)
nd.roi_heads.box_predictor.bbox_pred=nn.Linear(nd.roi_heads.box_predictor.bbox_pred.in_features,8)
nd.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/nodes_frcnn.pt',map_location=dev)); nd=nd.to(dev).eval()
lm=models.mobilenet_v3_small(weights=None); lm.classifier[3]=nn.Linear(lm.classifier[3].in_features,11); lm.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/landmark_mobilenet.pt',map_location=dev)); lm=lm.to(dev).eval()
rb=models.mobilenet_v3_small(weights=None); rb.classifier[3]=nn.Linear(rb.classifier[3].in_features,5); rb.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/robot_mobilenet.pt',map_location=dev)); rb=rb.to(dev).eval()
test_pred=json.loads(pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/test_missions_pred.json').read_text(encoding='utf-8'))
pred_by_idx={i:o for i,o in enumerate(test_pred)}
te_obs=json.loads((BASE/'test/observations.json').read_text(encoding='utf-8'))
import numpy as np
def assign_grid(pts):
    if len(pts)<10: return None
    xs=np.array([p[0] for p in pts]); ys=np.array([p[1] for p in pts])
    best=None
    for rows in range(5,10):
        for cols in range(5,10):
            ymin,ymax=ys.min(),ys.max(); xmin,xmax=xs.min(),xs.max()
            err=0; rcs=[]
            for x,y in pts:
                r=int(round((y-ymin)/max(1e-6,(ymax-ymin))*(rows-1))); c=int(round((x-xmin)/max(1e-6,(xmax-xmin))*(cols-1)))
                r=max(0,min(rows-1,r)); c=max(0,min(cols-1,c)); rcs.append((r,c))
                ey=ymin+(ymax-ymin)*r/(rows-1) if rows>1 else y; ex=xmin+(xmax-xmin)*c/(cols-1) if cols>1 else x
                err+=(x-ex)**2+(y-ey)**2
            err/=len(pts); dup=len(rcs)-len(set(rcs)); score=err+dup*5000
            if best is None or score<best[0]: best=(score,rows,cols,rcs)
    return best[1],best[2],best[3]
preds=[]
with torch.no_grad():
    for si in range(len(test_pred)):
        img_path=BASE/'test'/te_obs[si*10]['image']
        mission=test_pred[si]
        im=Image.open(img_path).convert('RGB'); w0,h0=im.size; sz=800
        imr=im.resize((sz,sz))
        out=nd([F.to_tensor(imr).to(dev)])[0]
        boxes=out['boxes'].cpu(); scores=out['scores'].cpu(); labels=out['labels'].cpu()
        pts=[]
        for i in range(len(boxes)):
            if labels[i]==1 and scores[i]>0.5:
                x0,y0,x1,y1=boxes[i].tolist()
                pts.append((((x0+x1)/2/sz*w0,(y0+y1)/2/sz*h0)))
        g=assign_grid(pts)
        if g is None:
            preds+=[2]*10; continue
        rows,cols,rcs=g
        # landmark/robot classify tung node crop (batch)
        crops=[]
        for (cx,cy) in pts:
            x0,y0=int(cx-32),int(cy-32)
            crops.append(TF(im.crop((max(0,x0),max(0,y0),min(w0,x0+64),min(h0,y0+64)))))
        import torch as T
        batch=T.stack(crops).to(dev)
        plm=[]; prb=[]
        for j in range(0,len(batch),64):
            b=batch[j:j+64]
            plm+=lm(b).argmax(1).cpu().tolist(); prb+=rb(b).argmax(1).cpu().tolist()
        # map rc -> landmark type / robot
        rc2lm={}; robot_rc=None; robot_hd='UP'
        for rc,li,ri in zip(rcs,plm,prb):
            if li<10: rc2lm.setdefault(rc,TYPES[li])
            if ri>0: robot_rc=rc; robot_hd=HEADS[ri-1]
        if robot_rc is None:
            preds+=[2]*10; continue
        # goal rc: cac node co landmark == goal (neu nhieu ban: chon gan robot nhat? + ref bac/nam? don gian: gan nhat)
        goal_nodes=[rc for rc,t in rc2lm.items() if t==mission['goal']]
        if not goal_nodes:
            preds+=[2]*10; continue
        # via? neu co via: target via truoc (don gian chi dung goal de greedy 1 buoc? se cai thien sau)
        # chon goal gan robot nhat (Manhattan)
        gr=min(goal_nodes,key=lambda rc:abs(rc[0]-robot_rc[0])+abs(rc[1]-robot_rc[1]))
        # valid moves: 4 huong ke trong grid + ton tai node (bo qua closed/oneway)
        rcset=set(rcs)
        for rid in range(10):
            # heading that? dung robot_hd that cho tat ca (BG: moi robot cung vi tri? dung vay)
            # greedy Manhattan toi gr
            best=None;bk=None
            for a,nxt in [(0,(robot_rc[0]-1,robot_rc[1])),(1,(robot_rc[0]+1,robot_rc[1])),(2,(robot_rc[0],robot_rc[1]-1)),(3,(robot_rc[0],robot_rc[1]+1))]:
                if nxt not in rcset: continue
                d=abs(nxt[0]-gr[0])+abs(nxt[1]-gr[1])
                k=(d,REL[(robot_hd,a)])
                if bk is None or k<bk: bk=k;best=a
            preds.append(int(best) if best is not None else 2)
        if (si+1)%100==0: print(f'done {si+1}/1200',flush=True)
json_str=json.dumps(preds)
pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/predictions_best.json').write_text(json_str)
print(f'saved predictions_best {len(preds)}')
