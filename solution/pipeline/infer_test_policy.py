"""Infer test voi full policies + PhoBERT + weather + detected graph (all-connected baseline)."""
import json, pathlib, collections, heapq
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
TFW=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
REL={('UP',0):0,('UP',3):1,('UP',2):2,('UP',1):3,('DOWN',1):0,('DOWN',2):1,('DOWN',3):2,('DOWN',0):3,('LEFT',2):0,('LEFT',0):1,('LEFT',1):2,('LEFT',3):3,('RIGHT',3):0,('RIGHT',1):1,('RIGHT',0):2,('RIGHT',2):3}
HEAD2DIR={'UP':0,'DOWN':1,'LEFT':2,'RIGHT':3}
def rc_to_action(a,b): return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((b[0]-a[0],b[1]-a[1]))
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('load models...',dev,flush=True)
nd=fasterrcnn_resnet50_fpn(weights=None)
nd.roi_heads.box_predictor.cls_score=nn.Linear(nd.roi_heads.box_predictor.cls_score.in_features,2)
nd.roi_heads.box_predictor.bbox_pred=nn.Linear(nd.roi_heads.box_predictor.bbox_pred.in_features,8)
nd.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/nodes_frcnn.pt',map_location=dev)); nd=nd.to(dev).eval()
lm=models.mobilenet_v3_small(weights=None); lm.classifier[3]=nn.Linear(lm.classifier[3].in_features,11); lm.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/landmark_mobilenet.pt',map_location=dev)); lm=lm.to(dev).eval()
rb=models.mobilenet_v3_small(weights=None); rb.classifier[3]=nn.Linear(rb.classifier[3].in_features,5); rb.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/robot_mobilenet.pt',map_location=dev)); rb=rb.to(dev).eval()
wx=models.mobilenet_v3_small(weights=None); wx.classifier[3]=nn.Linear(wx.classifier[3].in_features,2); wx.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/weather_img.pt',map_location=dev)); wx=wx.to(dev).eval()
test_pred=json.loads(pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/test_missions_pred.json').read_text(encoding='utf-8'))
te_obs=json.loads((BASE/'test/observations.json').read_text(encoding='utf-8'))
import numpy as np, sys
sys.path.insert(0,'/tmp/opencode')
from multileg import best_first_multileg
def cost_for(rid,urg,frag,wthr):
    if rid==0: return lambda e:1,{'S':0,'R':0,'L':0,'B':0},False
    if rid==1: return lambda e:1+(5 if e.get('status')=='crowded' else 0),{'S':0,'R':0,'L':0,'B':0},False
    if rid==2: return lambda e:0.5 if e.get('status')=='covered' else 1,{'S':0,'R':0,'L':0,'B':0},False
    if rid==3:
        if wthr=='rain': return lambda e:0.1 if e.get('status')=='covered' else 1,{'S':0,'R':0,'L':0,'B':0},False
        return lambda e:1+(2 if e.get('status')=='crowded' else 0),{'S':0,'R':0,'L':0,'B':0},False
    if rid==4: return lambda e:1+(1 if e.get('status')=='crowded' else 0),{'S':0,'R':0,'L':0,'B':0},True
    if rid==5: return lambda e:1,{'S':0,'R':1,'L':1,'B':20},False
    if rid==6:
        if urg: return lambda e:1,{'S':0,'R':0,'L':0,'B':0},False
        return lambda e:(0.7 if e.get('status')=='covered' else (6 if e.get('status')=='crowded' else 1)),{'S':0,'R':0,'L':0,'B':0},False
    if rid==7:
        if not frag: return lambda e:1,{'S':0,'R':0,'L':0,'B':0},False
        return lambda e:1+(5 if e.get('status')=='crowded' else 0),{'S':0,'R':1,'L':1,'B':10},False
    if rid==8: return lambda e:1,{'S':0,'R':0,'L':3,'B':6},False
    return None,None,None
def greedy9_man(rcset,robot_rc,head,tgt):
    best=None;bk=None
    for a,nxt in [(0,(robot_rc[0]-1,robot_rc[1])),(1,(robot_rc[0]+1,robot_rc[1])),(2,(robot_rc[0],robot_rc[1]-1)),(3,(robot_rc[0],robot_rc[1]+1))]:
        if nxt not in rcset: continue
        d=abs(nxt[0]-tgt[0])+abs(nxt[1]-tgt[1]);k=(d,REL[(head,a)])
        if bk is None or k<bk: bk=k;best=a
    return best
print('infer 1200...',flush=True)
preds=[]
with torch.no_grad():
    for si in range(len(test_pred)):
        img_path=BASE/'test'/te_obs[si*10]['image']; mission=test_pred[si]
        im=Image.open(img_path).convert('RGB'); w0,h0=im.size; sz=800; imr=im.resize((sz,sz))
        out=nd([F.to_tensor(imr).to(dev)])[0]
        boxes=out['boxes'].cpu(); scores=out['scores'].cpu(); labels=out['labels'].cpu()
        pts=[]
        for i in range(len(boxes)):
            if labels[i]==1 and scores[i]>0.5:
                x0,y0,x1,y1=boxes[i].tolist()
                pts.append(((x0+x1)/2/sz*w0,(y0+y1)/2/sz*h0))
        # grid assign (don gian nhu truoc)
        if len(pts)<10:
            preds+=[2]*10; continue
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
        rows,cols,rcs=best[1],best[2],best[3]
        # landmark/robot classify
        crops=[]
        for (cx,cy) in pts:
            x0,y0=int(cx-32),int(cy-32)
            crops.append(TF(im.crop((max(0,x0),max(0,y0),min(w0,x0+64),min(h0,y0+64)))))
        batch=torch.stack(crops).to(dev)
        plm=[]; prb=[]
        for j in range(0,len(batch),64):
            b=batch[j:j+64]
            plm+=lm(b).argmax(1).cpu().tolist(); prb+=rb(b).argmax(1).cpu().tolist()
        rc2lm={}; robot_rc=None; robot_hd='UP'
        for rc,li,ri in zip(rcs,plm,prb):
            if li<10: rc2lm.setdefault(rc,TYPES[li])
            if ri>0: robot_rc=rc; robot_hd=HEADS[ri-1]
        if robot_rc is None:
            preds+=[2]*10; continue
        # weather
        wpred=wx(TFW(im).unsqueeze(0).to(dev)).argmax(1).item()
        wthr='rain' if wpred==1 else 'dry'
        # goal/via rc (gan nhat, bo ref tam)
        goal_nodes=[rc for rc,t in rc2lm.items() if t==mission['goal']]
        if not goal_nodes:
            preds+=[2]*10; continue
        gr=min(goal_nodes,key=lambda rc:abs(rc[0]-robot_rc[0])+abs(rc[1]-robot_rc[1]))
        via_nodes=[]
        if mission['via']:
            via_nodes=[rc for rc,t in rc2lm.items() if t==mission['via']]
        # build fake scene cho multileg (all edges normal, no stairs/oneway/closed)
        rcset=set(rcs)
        edges=[]
        for r,c in rcset:
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb=(r+dr,c+dc)
                if nb in rcset:
                    # tranh lap: chi them 1 chieu (se xu ly 2 chieu trong build_adj? multileg build_adj tu edges list, can ca 2? No, build_adj them 2 chieu neu khong oneway. De don gian them 1 lan.)
                    pass
        # thay vi dung multileg voi edges that (can edges list), ta tu Dijkstra tren grid all-connected:
        # don gian: dung best_first_multileg voi fake scene co edges day du normal
        edge_list=[]
        seen=set()
        for r,c in rcset:
            for dr,dc,a in [(-1,0,0),(1,0,1),(0,-1,2),(0,1,3)]:
                nb=(r+dr,c+dc)
                if nb in rcset and ((nb,r,c) not in seen):
                    edge_list.append({'a':[r,c],'b':[nb[0],nb[1]],'status':'normal','stairs':False,'oneway_to':None})
                    seen.add((r,c,nb))
        fake={'nodes':[{'rc':list(rc),'xy':[0,0]} for rc in rcset],'edges':edge_list,'landmarks':[{'type':t,'rc':list(rc)} for rc,t in rc2lm.items()],'robot':{'rc':list(robot_rc),'heading':robot_hd},'weather':wthr,'mission':{'goal':mission['goal'],'goal_ref':None,'via':mission['via'],'via_ref':None,'urgent':mission['urgent'],'fragile':mission['fragile']}}
        for rid in range(10):
            if rid==9:
                # greedy toi via neu co else goal
                tgt=None
                if mission['via'] and via_nodes:
                    tgt=min(via_nodes,key=lambda rc:abs(rc[0]-robot_rc[0])+abs(rc[1]-robot_rc[1]))
                else: tgt=gr
                p=greedy9_man(rcset,robot_rc,robot_hd,tgt)
                preds.append(int(p) if p is not None else 2)
            else:
                fn,tc,leg=cost_for(rid,mission['urgent'],mission['fragile'],wthr)
                import sys; sys.path.insert(0,'/tmp/opencode')
                from multileg import best_first_multileg as BFM
                p=BFM(fake,fn,tc,leg)
                preds.append(int(p) if p is not None else 2)
        if (si+1)%200==0: print(f'done {si+1}/1200',flush=True)
pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/predictions_policy.json').write_text(json.dumps(preds))
print(f'saved predictions_policy {len(preds)}')
