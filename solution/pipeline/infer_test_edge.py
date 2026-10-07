"""Test full v2: nodes + landmark/robot/weather + EDGES status/stairs/oneway + policies + via.
Thay all-normal bang edge du doan (absolute, chua Siamese) de len 55-60%."""
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
TFW=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
STATUS=['normal','crowded','covered','closed']
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('load...',dev,flush=True)
nd=fasterrcnn_resnet50_fpn(weights=None)
nd.roi_heads.box_predictor.cls_score=nn.Linear(nd.roi_heads.box_predictor.cls_score.in_features,2)
nd.roi_heads.box_predictor.bbox_pred=nn.Linear(nd.roi_heads.box_predictor.bbox_pred.in_features,8)
nd.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/nodes_frcnn.pt',map_location=dev)); nd=nd.to(dev).eval()
lm=models.mobilenet_v3_small(weights=None); lm.classifier[3]=nn.Linear(lm.classifier[3].in_features,11); lm.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/landmark_mobilenet.pt',map_location=dev)); lm=lm.to(dev).eval()
rb=models.mobilenet_v3_small(weights=None); rb.classifier[3]=nn.Linear(rb.classifier[3].in_features,5); rb.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/robot_mobilenet.pt',map_location=dev)); rb=rb.to(dev).eval()
wx=models.mobilenet_v3_small(weights=None); wx.classifier[3]=nn.Linear(wx.classifier[3].in_features,2); wx.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/weather_img.pt',map_location=dev)); wx=wx.to(dev).eval()
ed=models.mobilenet_v3_small(weights=None); ed.classifier[3]=nn.Linear(ed.classifier[3].in_features,4); ed.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/edge_mobilenet.pt',map_location=dev)); ed=ed.to(dev).eval()
st=models.mobilenet_v3_small(weights=None); st.classifier[3]=nn.Linear(st.classifier[3].in_features,2); st.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/stairs_mobilenet.pt',map_location=dev)); st=st.to(dev).eval()
ow=models.mobilenet_v3_small(weights=None); ow.classifier[3]=nn.Linear(ow.classifier[3].in_features,2); ow.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/oneway_mobilenet.pt',map_location=dev)); ow=ow.to(dev).eval()
test_pred=json.loads(pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/test_missions_pred.json').read_text(encoding='utf-8'))
te_obs=json.loads((BASE/'test/observations.json').read_text(encoding='utf-8'))
import numpy as np, sys
sys.path.insert(0,'/tmp/opencode')
from multileg import best_first_multileg
REL={('UP',0):0,('UP',3):1,('UP',2):2,('UP',1):3,('DOWN',1):0,('DOWN',2):1,('DOWN',3):2,('DOWN',0):3,('LEFT',2):0,('LEFT',0):1,('LEFT',1):2,('LEFT',3):3,('RIGHT',3):0,('RIGHT',1):1,('RIGHT',0):2,('RIGHT',2):3}
def rc_to_action(a,b): return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((b[0]-a[0],b[1]-a[1]))
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
        # node landmark/robot
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
        wpred=wx(TFW(im).unsqueeze(0).to(dev)).argmax(1).item()
        wthr='rain' if wpred==1 else 'dry'
        # edges: grid neighbors + classify status/stairs (oneway tam bo qua huong, assume 2 chieu)
        rcset=set(rcs)
        # map rc -> center pixel de crop edge
        rc2xy={}
        for rc,(cx,cy) in zip(rcs,pts):
            rc2xy.setdefault(rc,(cx,cy))
        edge_list=[]
        pairs=[]
        for r,c in rcset:
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb=(r+dr,c+dc)
                if nb in rcset and (nb,r,c) not in [(p[1],p[0][0],p[0][1]) for p in pairs]:
                    pairs.append(((r,c),nb))
        if pairs:
            ecrops=[]
            for (a,b) in pairs:
                (x1,y1)=rc2xy[a]; (x2,y2)=rc2xy[b]
                mx,my=(x1+x2)/2,(y1+y2)/2
                ecrops.append(TF(im.crop((max(0,int(mx-32)),max(0,int(my-32)),min(w0,int(mx+32)),min(h0,int(my+32))))))
            eb=torch.stack(ecrops).to(dev)
            pstat=[]; pstairs=[]
            for j in range(0,len(eb),64):
                bb=eb[j:j+64]
                pstat+=ed(bb).argmax(1).cpu().tolist(); pstairs+=st(bb).argmax(1).cpu().tolist()
            for ((a,b),si_,sti) in zip(pairs,pstat,pstairs):
                # oneway: tam assume 2 chieu (se them oneway model + huong sau)
                edge_list.append({'a':list(a),'b':list(b),'status':STATUS[si_],'stairs':bool(sti),'oneway_to':None})
        fake={'nodes':[],'edges':edge_list,'landmarks':[{'type':t,'rc':list(rc)} for rc,t in rc2lm.items()],'robot':{'rc':list(robot_rc),'heading':robot_hd},'weather':wthr,'mission':{'goal':mission['goal'],'goal_ref':None,'via':mission['via'],'via_ref':None,'urgent':mission['urgent'],'fragile':mission['fragile']}}
        # R9 greedy can edge filtering (closed/stairs/oneway) - multileg tu xu ly closed/stairs/oneway qua build_adj
        for rid in range(10):
            if rid==9:
                # greedy manhattan voi edge loc (bo closed/stairs)
                rcset2=set(rcs)
                # loc: chi xet moves khong closed/stairs (oneway bo qua)
                # don gian: dung multileg base? Khong, greedy rieng:
                lms={}; 
                for rc,t in rc2lm.items():
                    lms.setdefault(t,[]).append(rc)
                tgt_list=lms.get(mission['via'] if mission['via'] else mission['goal'],[])
                if not tgt_list:
                    preds.append(2); continue
                # chon tgt gan robot nhat
                tgt=min(tgt_list,key=lambda rc:abs(rc[0]-robot_rc[0])+abs(rc[1]-robot_rc[1]))
                bestm=None;bk=None
                for a,nxt in [(0,(robot_rc[0]-1,robot_rc[1])),(1,(robot_rc[0]+1,robot_rc[1])),(2,(robot_rc[0],robot_rc[1]-1)),(3,(robot_rc[0],robot_rc[1]+1))]:
                    if nxt not in rcset2: continue
                    # kiem tra edge closed/stairs?
                    # tim edge
                    found=None
                    for e in edge_list:
                        if (tuple(e['a'])==robot_rc and tuple(e['b'])==nxt) or (tuple(e['b'])==robot_rc and tuple(e['a'])==nxt):
                            found=e; break
                    if found and (found['status']=='closed' or found['stairs']): continue
                    d=abs(nxt[0]-tgt[0])+abs(nxt[1]-tgt[1]);k=(d,REL[(robot_hd,a)])
                    if bk is None or k<bk: bk=k;bestm=a
                preds.append(int(bestm) if bestm is not None else 2)
            else:
                fn,tc,leg=cost_for(rid,mission['urgent'],mission['fragile'],wthr)
                p=best_first_multileg(fake,fn,tc,leg)
                preds.append(int(p) if p is not None else 2)
        if (si+1)%200==0: print(f'done {si+1}/1200',flush=True)
pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/predictions_edge.json').write_text(json.dumps(preds))
print(f'saved predictions_edge {len(preds)}')
