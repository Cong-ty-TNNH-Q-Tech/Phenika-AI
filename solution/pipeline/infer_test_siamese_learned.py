"""Siamese full: legend swatches + edge mean-RGB matching + policies. 1200 test."""
import json, pathlib, collections
import torch
from PIL import Image
import torch.nn as nn
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import models, transforms
from torchvision.transforms import functional as F
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
KINDS=['normal','crowded','covered','closed','stairs','oneway']
TYPES=['library','dorm','sports','clinic','canteen','parking','lecture','lab','office','gate']
HEADS=['UP','DOWN','LEFT','RIGHT']
TF=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
TFW=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('load...',dev,flush=True)
nd=fasterrcnn_resnet50_fpn(weights=None)
nd.roi_heads.box_predictor.cls_score=nn.Linear(nd.roi_heads.box_predictor.cls_score.in_features,2)
nd.roi_heads.box_predictor.bbox_pred=nn.Linear(nd.roi_heads.box_predictor.bbox_pred.in_features,8)
nd.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/nodes_frcnn.pt',map_location=dev)); nd=nd.to(dev).eval()
lg=fasterrcnn_resnet50_fpn(weights=None)
lg.roi_heads.box_predictor.cls_score=nn.Linear(lg.roi_heads.box_predictor.cls_score.in_features,7)
lg.roi_heads.box_predictor.bbox_pred=nn.Linear(lg.roi_heads.box_predictor.bbox_pred.in_features,28)
lg.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/legend_frcnn.pt',map_location=dev)); lg=lg.to(dev).eval()
lm=models.mobilenet_v3_small(weights=None); lm.classifier[3]=nn.Linear(lm.classifier[3].in_features,11); lm.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/landmark_mobilenet.pt',map_location=dev)); lm=lm.to(dev).eval()
rb=models.mobilenet_v3_small(weights=None); rb.classifier[3]=nn.Linear(rb.classifier[3].in_features,5); rb.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/robot_mobilenet.pt',map_location=dev)); rb=rb.to(dev).eval()
wx=models.mobilenet_v3_small(weights=None); wx.classifier[3]=nn.Linear(wx.classifier[3].in_features,2); wx.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/weather_img.pt',map_location=dev)); wx=wx.to(dev).eval()
st=models.mobilenet_v3_small(weights=None); st.classifier[3]=nn.Linear(st.classifier[3].in_features,2); st.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/stairs_mobilenet.pt',map_location=dev)); st=st.to(dev).eval()
owd=models.mobilenet_v3_small(weights=None); owd.classifier[3]=nn.Linear(owd.classifier[3].in_features,3); owd.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/oneway_dir.pt',map_location=dev)); owd=owd.to(dev).eval()
TF128=transforms.Compose([transforms.Resize((64,128)),transforms.ToTensor()])
import sys as _sys; _sys.path.insert(0,'/mnt/hdd2/qtech/Phenika-AI/solution/cv')
from train_siamese import Siam2
siam=Siam2().to(dev); siam.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/siamese_mlp.pt',map_location=dev)); siam=siam.to(dev).eval()
ROAD_ORDER=['normal','crowded','covered','closed']
test_pred=json.loads(pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/test_missions_pred.json').read_text(encoding='utf-8'))
test_gref=json.loads(pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/test_gref_pred.json').read_text(encoding='utf-8'))
te_obs=json.loads((BASE/'test/observations.json').read_text(encoding='utf-8'))
def resolve_ref_nsew(goal_nodes, kind):
    # goal_nodes: list rc; kind north/south/east/west ( Bac=tren=min row)
    if not goal_nodes or kind not in ('north','south','east','west'): return None
    if kind=='north': return min(goal_nodes, key=lambda rc:(rc[0],rc[1]))
    if kind=='south': return max(goal_nodes, key=lambda rc:(rc[0],rc[1]))
    if kind=='west': return min(goal_nodes, key=lambda rc:(rc[1],rc[0]))
    if kind=='east': return max(goal_nodes, key=lambda rc:(rc[1],rc[0]))
    return None
import sys; sys.path.insert(0,'/tmp/opencode')
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
def mean_rgb(im):
    import numpy as np
    a=np.array(im.resize((32,32))).reshape(-1,3).mean(0)
    return a
preds=[]
with torch.no_grad():
    for si in range(len(test_pred)):
        img_path=BASE/'test'/te_obs[si*10]['image']; mission=test_pred[si]
        im=Image.open(img_path).convert('RGB'); w0,h0=im.size; sz=800; imr=im.resize((sz,sz))
        # nodes
        out=nd([F.to_tensor(imr).to(dev)])[0]
        boxes=out['boxes'].cpu(); scores=out['scores'].cpu(); labels=out['labels'].cpu()
        pts=[]
        for i in range(len(boxes)):
            if labels[i]==1 and scores[i]>0.5:
                x0,y0,x1,y1=boxes[i].tolist()
                pts.append(((x0+x1)/2/sz*w0,(y0+y1)/2/sz*h0))
        # legend swatches
        outl=lg([F.to_tensor(imr).to(dev)])[0]
        bl=outl['boxes'].cpu(); sl=outl['scores'].cpu(); ll=outl['labels'].cpu()
        sw_mean={}; sw_tensors={}
        for i in range(len(bl)):
            if sl[i]>0.5 and 1<=ll[i]<=6:
                k=KINDS[ll[i]-1]
                if k in ('normal','crowded','covered','closed') and k not in sw_mean:
                    x0,y0,x1,y1=bl[i].tolist()
                    x0,y0,x1,y1=x0/sz*w0,y0/sz*h0,x1/sz*w0,y1/sz*h0
                    crop=im.crop((max(0,int(x0)),max(0,int(y0)),min(w0,int(x1)),min(h0,int(y1))))
                    if crop.size[0]>5 and crop.size[1]>5:
                        sw_mean[k]=mean_rgb(crop); sw_tensors[k]=TF(crop)
        if len(pts)<10 or len(sw_mean)<3:
            preds+=[2]*10; continue
        # grid
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
        # landmark/robot
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
        # edges: Siamese mean-RGB to swatches + stairs classifier
        rcset=set(rcs)
        rc2xy={}
        for rc,(cx,cy) in zip(rcs,pts):
            rc2xy.setdefault(rc,(cx,cy))
        pairs=[]
        for r,c in rcset:
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb=(r+dr,c+dc)
                if nb in rcset:
                    a,b=sorted([(r,c),nb])
                    if (a,b) in [(p[0],p[1]) for p in pairs]: continue
                    pairs.append((a,b))
        edge_list=[]
        if pairs:
            ecrops=[]; epts=[]
            for (a,b) in pairs:
                (x1,y1)=rc2xy[a]; (x2,y2)=rc2xy[b]
                mx,my=(x1+x2)/2,(y1+y2)/2
                crop=im.crop((max(0,int(mx-32)),max(0,int(my-32)),min(w0,int(mx+32)),min(h0,int(my+32))))
                ecrops.append(TF(crop)); epts.append((crop,a,b))
            eb=torch.stack(ecrops).to(dev)
            pstairs=[]
            for j in range(0,len(eb),64):
                pstairs+=st(eb[j:j+64]).argmax(1).cpu().tolist()
            # oneway direction (a<b sorted, 0=2chieu 1=a->b 2=b->a)
            owcrops=[]
            for (a,b) in pairs:
                (x1,y1)=rc2xy[a]; (x2,y2)=rc2xy[b]
                mx,my=(x1+x2)/2,(y1+y2)/2
                owcrops.append(TF128(im.crop((max(0,int(mx-32)),max(0,int(my-64)),min(w0,int(mx+32)),min(h0,int(my+64))))))
            ob=torch.stack(owcrops).to(dev)
            pow_=[]
            for j in range(0,len(ob),64):
                pow_+=owd(ob[j:j+64]).argmax(1).cpu().tolist()
            # Learned Siamese: neu du 4 swatches thi dung MLP, else fallback meanRGB
            use_learned = all(k in sw_tensors for k in ROAD_ORDER)
            if use_learned:
                sw_batch = torch.stack([sw_tensors[k] for k in ROAD_ORDER]).to(dev)  # 4x3x64x64
                pstatus = []
                for j in range(0, len(eb), 32):
                    be = eb[j:j+32]
                    sws = sw_batch.unsqueeze(0).repeat(len(be), 1, 1, 1, 1)
                    pstatus += siam(be, sws).argmax(1).cpu().tolist()
                for ((crop, a, b), sti, psi, pows) in zip(epts, pstairs, pstatus, pow_):
                    ow_to = list(b) if pows==1 else (list(a) if pows==2 else None)
                    edge_list.append({'a': list(a), 'b': list(b), 'status': ROAD_ORDER[psi], 'stairs': bool(sti), 'oneway_to': ow_to})
            else:
                for (crop,a,b),sti in zip(epts,pstairs):
                    # fallback mean RGB (oneway assume 2 chieu)
                    em=mean_rgb(crop)
                    bestk=None; bestd=None
                    for k,sm in sw_mean.items():
                        d=float(((em-sm)**2).sum())
                        if bestd is None or d<bestd: bestd=d; bestk=k
                    edge_list.append({'a':list(a),'b':list(b),'status':bestk or 'normal','stairs':bool(sti),'oneway_to':None})
        # ref N/S/E/W: loc goal dups theo kind (near/far/NONE giu min-cost)
        gkind = test_gref[si] if si < len(test_gref) else 'NONE'
        _rc2lm = dict(rc2lm)
        _goal_nodes = [rc for rc,t in _rc2lm.items() if t==mission['goal']]
        _resolved = resolve_ref_nsew(_goal_nodes, gkind)
        if _resolved is not None:
            _rc2lm = {rc:(mission['goal'] if rc==_resolved else t) if t==mission['goal'] else t for rc,t in _rc2lm.items()}
            # xoa cac ban goal khac ref (giu duy nhat resolved)
            _rc2lm = {rc:t for rc,t in _rc2lm.items() if not (t==mission['goal'] and rc!=_resolved)}
            rc2lm = _rc2lm
        fake={'nodes':[],'edges':edge_list,'landmarks':[{'type':t,'rc':list(rc)} for rc,t in rc2lm.items()],'robot':{'rc':list(robot_rc),'heading':robot_hd},'weather':wthr,'mission':{'goal':mission['goal'],'goal_ref':None,'via':mission['via'],'via_ref':None,'urgent':mission['urgent'],'fragile':mission['fragile']}}
        for rid in range(10):
            if rid==9:
                # greedy voi loc closed/stairs
                lms2={}
                for rc,t in rc2lm.items(): lms2.setdefault(t,[]).append(rc)
                tgt_list=lms2.get(mission['via'] if mission['via'] else mission['goal'],[])
                if not tgt_list:
                    preds.append(2); continue
                tgt=min(tgt_list,key=lambda rc:abs(rc[0]-robot_rc[0])+abs(rc[1]-robot_rc[1]))
                bestm=None;bk=None
                for a,nxt in [(0,(robot_rc[0]-1,robot_rc[1])),(1,(robot_rc[0]+1,robot_rc[1])),(2,(robot_rc[0],robot_rc[1]-1)),(3,(robot_rc[0],robot_rc[1]+1))]:
                    if nxt not in rcset: continue
                    found=None
                    for e in edge_list:
                        if (tuple(e['a'])==robot_rc and tuple(e['b'])==nxt) or (tuple(e['b'])==robot_rc and tuple(e['a'])==nxt):
                            found=e; break
                    if found and (found['status']=='closed' or found['stairs']): continue
                    if found and found.get('oneway_to') is not None and tuple(found['oneway_to'])!=nxt: continue
                    d=abs(nxt[0]-tgt[0])+abs(nxt[1]-tgt[1]);k=(d,REL[(robot_hd,a)])
                    if bk is None or k<bk: bk=k;bestm=a
                preds.append(int(bestm) if bestm is not None else 2)
            else:
                fn,tc,leg=cost_for(rid,mission['urgent'],mission['fragile'],wthr)
                p=best_first_multileg(fake,fn,tc,leg)
                preds.append(int(p) if p is not None else 2)
        if (si+1)%200==0: print(f'done {si+1}/1200',flush=True)
pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/predictions_full.json').write_text(json.dumps(preds))
print(f'saved full-oneway-ref {len(preds)}')
