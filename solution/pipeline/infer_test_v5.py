"""Siamese full: legend swatches + edge mean-RGB matching + policies. 1200 test."""
import json, pathlib, collections
import torch
from PIL import Image
import torch.nn as nn
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import models, transforms
from torchvision.transforms import functional as F
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/v3data/delivery_public')
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
lm=models.resnet18(weights=None); lm.fc=nn.Linear(lm.fc.in_features,11); lm.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/landmark_resnet18.pt',map_location=dev)); lm=lm.to(dev).eval()
rb=models.mobilenet_v3_small(weights=None); rb.classifier[3]=nn.Linear(rb.classifier[3].in_features,5); rb.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/robot_mobilenet.pt',map_location=dev)); rb=rb.to(dev).eval()
wx=models.mobilenet_v3_small(weights=None); wx.classifier[3]=nn.Linear(wx.classifier[3].in_features,2); wx.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/weather_img.pt',map_location=dev)); wx=wx.to(dev).eval()
st=models.mobilenet_v3_small(weights=None); st.classifier[3]=nn.Linear(st.classifier[3].in_features,2); st.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/stairs_mobilenet.pt',map_location=dev)); st=st.to(dev).eval()
owd=models.mobilenet_v3_small(weights=None); owd.classifier[3]=nn.Linear(owd.classifier[3].in_features,3); owd.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/oneway_dir.pt',map_location=dev)); owd=owd.to(dev).eval()
edg=models.resnet18(weights=None); edg.fc=nn.Linear(edg.fc.in_features,4); edg.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/edge_resnet18.pt',map_location=dev)); edg=edg.to(dev).eval()
pres=models.resnet18(weights=None); pres.fc=nn.Linear(pres.fc.in_features,2); pres.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/edge_presence.pt',map_location=dev)); pres=pres.to(dev).eval()
TFEN=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
TFL=TFEN
TF128=transforms.Compose([transforms.Resize((64,128)),transforms.ToTensor()])
import sys as _sys; _sys.path.insert(0,'/mnt/hdd2/qtech/Phenika-AI/solution/cv')
from train_siamese import Siam2
siam=Siam2().to(dev); siam.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/siamese_mlp.pt',map_location=dev)); siam=siam.to(dev).eval()
ROAD_ORDER=['normal','crowded','covered','closed']
test_pred=json.loads(pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/v3_test_missions.json').read_text(encoding='utf-8'))
import sys as _nlp_sys; _nlp_sys.path.insert(0,'/mnt/hdd2/qtech/Phenika-AI/solution/nlp')
from nlp_rules import rule_kind as _rule_kind, match_anchor as _match_anchor
te_obs0=json.loads((BASE/'test/observations.json').read_text(encoding='utf-8'))
for _i,_m in enumerate(test_pred):
    _rk=_rule_kind(te_obs0[_i*10]['mission'])
    if _rk is not None: _m['gref_kind']=_rk
    _m['mission_text']=te_obs0[_i*10]['mission']
test_gref=[]
te_obs=json.loads((BASE/'test/observations.json').read_text(encoding='utf-8'))
# no test labels
def resolve_ref_nsew(goal_nodes, kind):
    # goal_nodes: list rc; kind north/south/east/west ( Bac=tren=min row)
    if not goal_nodes or kind not in ('north','south','east','west'): return None
    if kind=='north': return min(goal_nodes, key=lambda rc:(rc[0],rc[1]))
    if kind=='south': return max(goal_nodes, key=lambda rc:(rc[0],rc[1]))
    if kind=='west': return min(goal_nodes, key=lambda rc:(rc[1],rc[0]))
    if kind=='east': return max(goal_nodes, key=lambda rc:(rc[1],rc[0]))
    return None
import unicodedata as _ud
def _strip(s): return ''.join(c for c in _ud.normalize('NFD',s) if _ud.category(c)!='Mn').lower()
_ALIASES={
 'library':['thu vien','noi muon giao trinh','cho tra sach','phong doc','tra sach','muon sach','khu doc sach','noi muon sach tham khao','the thu vien'],
 'dorm':['ky tuc xa','ktx','khu noi tru','toa nha o cua sinh vien','phong o sinh vien','nha o sinh vien','day phong noi tru','khu o cua sinh vien','noi noi tru'],
 'sports':['nha thi dau','san tap','san bong','doi bong','nha the chat','san luyen tap','san the thao','khu the thao','nha dau'],
 'clinic':['tram y te','tram xa','phong kham','phong y te','tram y','noi kham suc khoe','noi kham'],
 'canteen':['can tin','cang tin','nha an','bep an','bep truong','khu an uong','phong an tap the','noi phuc vu bua trua','phong an'],
 'parking':['bai do xe','khu gui xe','nha xe','bai xe','cho do xe','khu dau xe','ham xe','bai dau xe'],
 'lecture':['phong hoc lon','khu giang duong','hoi truong hoc','lop hoc','giang duong','phong hoc','day phong hoc','noi len lop','toa giang duong','toa nha giang duong'],
 'lab':['phong lab','phong thi nghiem','xuong thuc hanh','lab hoa','phong thuc nghiem','phong thuc hanh','noi lam thi nghiem','noi thi nghiem'],
 'office':['phong mot cua','toa hieu bo','khu hanh chinh','van phong khoa','hieu bo','phong giao vu','noi nop ho so','phong cong tac','van phong'],
 'gate':['cong truong','cong vao','loi vao truong','cong chinh','chot cong','chot bao ve cong','loi vao','bao ve cong'],
}
for _k in _ALIASES: _ALIASES[_k]=sorted(set(_strip(a) for a in _ALIASES[_k]),key=len,reverse=True)
_DIRS=[('phia tren ban do','north'),('man tren','north'),('phia tren','north'),('ben tren','north'),('phia bac','north'),('man bac','north'),('ban do phia bac','north'),
 ('phia duoi ban do','south'),('man duoi','south'),('phia duoi','south'),('ben duoi','south'),('phia nam','south'),('man nam','south'),
 ('ben trai ban do','west'),('man trai','west'),('phia trai','west'),('ben trai','west'),('phia tay','west'),('man tay','west'),
 ('ben phai ban do','east'),('man phai','east'),('phia phai','east'),('ben phai','east'),('phia dong','east'),('man dong','east')]
def extract_goal_nsew(text, goal):
    t=_strip(text); gpos=None
    for a in _ALIASES.get(goal,[]):
        p=t.find(a)
        if p>=0: gpos=p; break
    cands=[]
    for w,kind in _DIRS:
        p=t.find(w)
        if p>=0: cands.append((p,kind))
    if gpos is not None and cands: return min(cands,key=lambda pc:abs(pc[0]-gpos))[1]
    if cands: return max(cands,key=lambda pc:pc[0])[1]
    return None
def extract_nearfar_kind(text, target):
    t=_strip(text); tpos=None
    for a in _ALIASES.get(target,[]):
        p=t.find(a)
        if p>=0: tpos=p; break
    if tpos is None: return None
    near_p=far_p=1e9
    for w in ['gan','sat','khu gan','gan sat']:
        p=t.find(w)
        if p>=0: near_p=min(near_p,abs(p-tpos))
    for w in ['xa','cach xa','xa xa']:
        p=t.find(w)
        if p>=0: far_p=min(far_p,abs(p-tpos))
    if near_p==far_p: return None
    return 'near' if near_p<far_p else 'far'
import sys as _sys2; _sys2.path.insert(0,'/mnt/hdd2/qtech/Phenika-AI/solution/nlp')
from anchor_rule_v2 import predict_anchor as _predict_anchor
def resolve_ref_nearfar(goal_nodes, kind, text, goal, via, landmarks):
    # landmarks: list dict {type,rc} (detected). anchor phai single.
    if not goal_nodes or kind not in ('near','far') or len(goal_nodes)<2: return None
    anc = _predict_anchor(text, goal, via, landmarks)
    if not anc: return None
    anc_nodes=[lm['rc'] for lm in landmarks if lm['type']==anc]
    if len(anc_nodes)!=1: return None
    ax,ay=anc_nodes[0][0],anc_nodes[0][1]
    import math
    if kind=='near': return min(goal_nodes,key=lambda rc:math.hypot(rc[0]-ax,rc[1]-ay))
    return max(goal_nodes,key=lambda rc:math.hypot(rc[0]-ax,rc[1]-ay))
import sys; sys.path.insert(0,'/tmp/opencode')
from multileg import best_first_multileg
import sys as _ps; _ps.path.insert(0,'/mnt/hdd2/qtech/Phenika-AI/solution/oracle')
import policy_v5 as _P5
REL={('UP',0):0,('UP',3):1,('UP',2):2,('UP',1):3,('DOWN',1):0,('DOWN',2):1,('DOWN',3):2,('DOWN',0):3,('LEFT',2):0,('LEFT',0):1,('LEFT',1):2,('LEFT',3):3,('RIGHT',3):0,('RIGHT',1):1,('RIGHT',0):2,('RIGHT',2):3}
def rc_to_action(a,b): return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((b[0]-a[0],b[1]-a[1]))
def cost_for(rid,urg,frag,wthr):
    if rid==0: return lambda e:1,{'S':0,'R':0,'L':0,'B':0},False
    if rid==1: return lambda e:1+(5 if e.get('status')=='crowded' else 0),{'S':0,'R':0,'L':0,'B':0},False
    if rid==2: return lambda e:0.5 if e.get('status')=='covered' else 1,{'S':0,'R':0,'L':0,'B':0},False
    if rid==3:
        if wthr=='rain': return lambda e:0.3 if e.get('status')=='covered' else 1,{'S':0,'R':0,'L':0,'B':0},False
        return lambda e:1+(2 if e.get('status')=='crowded' else 0),{'S':0,'R':0,'L':0,'B':0},False
    if rid==4: return lambda e:1+(1 if e.get('status')=='crowded' else 0),{'S':0,'R':0,'L':0,'B':0},True
    if rid==5: return lambda e:1,{'S':0,'R':2,'L':2,'B':20},False
    if rid==6:
        if urg: return lambda e:1,{'S':0,'R':0,'L':0,'B':0},False
        return lambda e:1+(5 if e.get('status')=='crowded' else 0),{'S':0,'R':0,'L':0,'B':0},False
    if rid==7:
        if not frag: return lambda e:1,{'S':0,'R':0,'L':0,'B':0},False
        return lambda e:1+(5 if e.get('status')=='crowded' else 0),{'S':0,'R':1,'L':1,'B':10},False
    if rid==8: return lambda e:1,{'S':0,'R':0.5,'L':5,'B':10},False
    return None,None,None
def mean_rgb(im):
    import numpy as np
    a=np.array(im.resize((32,32))).reshape(-1,3).mean(0)
    return a
def assign_grid_affine(pts):
    import numpy as _np
    P=_np.array(pts,dtype=float); N=len(P); best=None
    for R in range(5,10):
        for C in range(5,10):
            r=_np.clip(_np.round((P[:,1]-P[:,1].min())/max(1e-6,(P[:,1].max()-P[:,1].min()))*(R-1)),0,R-1).astype(int)
            c=_np.clip(_np.round((P[:,0]-P[:,0].min())/max(1e-6,(P[:,0].max()-P[:,0].min()))*(C-1)),0,C-1).astype(int)
            for _ in range(6):
                A=_np.stack([_np.ones(N),r,c],1); X,_,_,_=_np.linalg.lstsq(A,P,rcond=None)
                O=X[0]; u=X[1]; v=X[2]; M=_np.array([[u[0],v[0]],[u[1],v[1]]]); Minv=_np.linalg.pinv(M)
                rc=((P-O)@Minv.T); r=_np.clip(_np.round(rc[:,0]),0,R-1).astype(int); c=_np.clip(_np.round(rc[:,1]),0,C-1).astype(int)
            pred=A@X; err=float(((pred-P)**2).sum(1).mean()); rcs=list(zip(r.tolist(),c.tolist()))
            dup=len(rcs)-len(set(rcs)); score=err+dup*5000
            if best is None or score<best[0]: best=(score,R,C,rcs)
    return best[1],best[2],best[3]
preds=[]
from collections import Counter as _C
REASON=_C()
with torch.no_grad():
    for si in range(len(test_pred)):
        img_path=BASE/'test'/te_obs[si*10]['image']; mission=test_pred[si]
        im=Image.open(img_path).convert('RGB'); w0,h0=im.size; sz=1120; imr=im.resize((sz,sz))
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
            REASON['nodes<10' if len(pts)<10 else 'swatch<3']+=1
            preds+=[2]*10; continue
        # grid (affine lattice fit, robust hon linear)
        rows,cols,rcs=assign_grid_affine(pts)
        # landmark/robot
        crops=[]
        crops_rb=[]
        for (cx,cy) in pts:
            x0,y0=int(cx-32),int(cy-32)
            crops.append(TFL(im.crop((max(0,x0),max(0,y0),min(w0,x0+64),min(h0,y0+64)))))
            crops_rb.append(TF(im.crop((max(0,x0),max(0,y0),min(w0,x0+64),min(h0,y0+64)))))
        batch=torch.stack(crops).to(dev)
        batch_rb=torch.stack(crops_rb).to(dev)
        plm=[]; prb=[]; plm_soft=[]
        for j in range(0,len(batch),64):
            b=batch[j:j+64]
            lo=lm(b); plm+=lo.argmax(1).cpu().tolist(); plm_soft+=torch.softmax(lo,1).cpu().tolist()
            prb+=rb(batch_rb[j:j+64]).argmax(1).cpu().tolist()
        rc2lm={}; robot_rc=None; robot_hd='UP'
        for rc,li,ri in zip(rcs,plm,prb):
            if li<10: rc2lm.setdefault(rc,TYPES[li])
            if ri>0: robot_rc=rc; robot_hd=HEADS[ri-1]
        if robot_rc is None:
            REASON['no_robot']+=1
            preds+=[2]*10; continue
        # softmax fallback: neu goal/via type vang mat, gan node co prob cao nhat
        _TI={t:i for i,t in enumerate(TYPES)}
        for _need in [mission['goal'], mission['via']]:
            if _need and _need in _TI and _need not in rc2lm.values():
                _bi=max(range(len(rcs)), key=lambda k: plm_soft[k][_TI[_need]])
                rc2lm[rcs[_bi]]=_need
        # improved anchor with detected landmarks
        if mission.get('gref_kind') in ('anchor_near','near','far'):
            _det_lms=[{'type':t,'rc':list(rc)} for rc,t in rc2lm.items()]
            _anc=_match_anchor(mission.get('mission_text',''),mission['goal'],mission['via'],_det_lms)
            if _anc is not None: mission['gref_anchor']=_anc
        wpred=wx(TFW(im).unsqueeze(0).to(dev)).argmax(1).item()
        wthr='rain' if wpred==1 else 'dry'
        import numpy as _np2; _style='night' if float(_np2.array(im.resize((64,64))).mean())<100 else 'other'
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
            # abs ResNet18 (closed detector 99%) on normalized crops
            ecrops_n=[TFEN(im.crop((max(0,int((rc2xy[a][0]+rc2xy[b][0])/2-32)),max(0,int((rc2xy[a][1]+rc2xy[b][1])/2-32)),min(w0,int((rc2xy[a][0]+rc2xy[b][0])/2+32)),min(h0,int((rc2xy[a][1]+rc2xy[b][1])/2+32))))) for (a,b) in pairs]
            eb_n=torch.stack(ecrops_n).to(dev)
            pabs=[]
            for j in range(0,len(eb_n),128): pabs+=edg(eb_n[j:j+128]).argmax(1).cpu().tolist()
            ppres=[]
            for j in range(0,len(eb_n),128): ppres+=pres(eb_n[j:j+128]).argmax(1).cpu().tolist()
            ROAD_ORDER=['normal','crowded','covered','closed']
            p_siam=None
            if all(k in sw_tensors for k in ROAD_ORDER):
                sw_batch=torch.stack([sw_tensors[k] for k in ROAD_ORDER]).to(dev)
                p_siam=[]
                for j in range(0,len(eb),32):
                    be=eb[j:j+32]; sws=sw_batch.unsqueeze(0).repeat(len(be),1,1,1,1)
                    lo=siam(be,sws); lo=lo.clone(); lo[:,3]=-1e9
                    p_siam+=lo.argmax(1).cpu().tolist()
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
            # HYBRID: abs closed (99%); non-closed -> meanRGB swatch trong {normal,crowded,covered}
            for _i,((crop, a, b), sti, pows, pai, prs) in enumerate(zip(epts, pstairs, pow_, pabs, ppres)):
                if prs==0: continue
                ow_to = list(b) if pows==1 else (list(a) if pows==2 else None)
                if pai==3:
                    status='closed'
                elif p_siam is not None:
                    status=ROAD_ORDER[p_siam[_i]]
                else:
                    em=mean_rgb(crop); bestk=None; bestd=None
                    for k in ('normal','crowded','covered'):
                        sm=sw_mean.get(k)
                        if sm is None: continue
                        d=float(((em-sm)**2).sum())
                        if bestd is None or d<bestd: bestd=d; bestk=k
                    status=bestk or 'normal'
                edge_list.append({'a': list(a), 'b': list(b), 'status': status, 'stairs': bool(sti), 'oneway_to': ow_to})
        # v3 ref resolution (12 kinds)
        import math as _math
        gk = mission.get('gref_kind','NONE'); ga = mission.get('gref_anchor')
        _rc2lm = dict(rc2lm)
        _lms = [(t, rc) for rc,t in _rc2lm.items()]
        goal_type = mission['goal']; goal_resolved = None
        if gk in ('north_most','south_most','west_most','east_most'):
            kf = {'north_most':(lambda rc:(rc[0],rc[1])), 'south_most':(lambda rc:(-rc[0],rc[1])),
                  'west_most':(lambda rc:(rc[1],rc[0])), 'east_most':(lambda rc:(-rc[1],rc[0]))}[gk]
            if _lms:
                bt, brc = min(_lms, key=lambda x: kf(x[1]))
                goal_type = bt; goal_resolved = brc
        elif gk == 'anchor_near' and ga is not None:
            anc = [rc for t,rc in _lms if t==ga]
            if len(anc)==1:
                ax,ay = anc[0]
                others = [(t,rc) for t,rc in _lms if t!=ga]
                if others:
                    bt,brc = min(others, key=lambda x: _math.hypot(x[1][0]-ax, x[1][1]-ay))
                    goal_type = bt; goal_resolved = brc
        elif gk in ('north','south','west','east'):
            gn = [rc for t,rc in _lms if t==goal_type]
            if gn:
                kf = {'north':(lambda rc:(rc[0],rc[1])), 'south':(lambda rc:(-rc[0],rc[1])),
                      'west':(lambda rc:(rc[1],rc[0])), 'east':(lambda rc:(-rc[1],rc[0]))}[gk]
                goal_resolved = min(gn, key=kf)
        elif gk in ('near','far') and ga is not None:
            gn = [rc for t,rc in _lms if t==goal_type]
            anc = [rc for t,rc in _lms if t==ga]
            if len(anc)==1 and len(gn)>=2:
                ax,ay = anc[0]
                f = (lambda rc: _math.hypot(rc[0]-ax,rc[1]-ay)) if gk=='near' else (lambda rc: -_math.hypot(rc[0]-ax,rc[1]-ay))
                goal_resolved = min(gn, key=f)
        if goal_resolved is not None:
            _rc2lm = {rc:t for rc,t in _rc2lm.items() if not (t==goal_type and rc!=goal_resolved)}
            rc2lm = _rc2lm
        # via ref (v3: near/far/north/south/west/east)
        via_type = mission['via']
        if via_type is not None:
            _via_nodes = [rc for t,rc in _rc2lm.items() if t==via_type]
            if len(_via_nodes)>=2:
                _vres = None
                _vrk = mission.get('via_ref_kind')
                if _vrk in ('north','south','west','east'):
                    kf = {'north':(lambda rc:(rc[0],rc[1])), 'south':(lambda rc:(-rc[0],rc[1])),
                          'west':(lambda rc:(rc[1],rc[0])), 'east':(lambda rc:(-rc[1],rc[0]))}[_vrk]
                    _vres = min(_via_nodes, key=kf)
                if _vres is not None:
                    _rc2lm = {rc:t for rc,t in _rc2lm.items() if not (t==via_type and rc!=_vres)}
                    rc2lm = _rc2lm
        fake={'nodes':[],'edges':edge_list,'landmarks':[{'type':t,'rc':list(rc)} for rc,t in rc2lm.items()],'robot':{'rc':list(robot_rc),'heading':robot_hd},'weather':wthr,'style':_style,'mission':{'goal':goal_type,'goal_ref':None,'via':via_type,'via_ref':None,'urgent':mission['urgent'],'fragile':mission['fragile']}}
        _P5preds = _P5.predict_all(fake); _P5g9 = _P5.greedy9(fake)
        for rid in range(10):
            if rid==9:
                p = _P5g9
                if p is None:
                    # greedy fallback voi loc closed/stairs
                    lms2={}
                    for rc,t in rc2lm.items(): lms2.setdefault(t,[]).append(rc)
                    tgt_list=lms2.get(via_type if via_type else goal_type,[])
                    if tgt_list:
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
                            ec=(1 if (found and found.get('status')=='crowded') else 0)
                            d=abs(nxt[0]-tgt[0])+abs(nxt[1]-tgt[1]);k=(d+ec,REL[(robot_hd,a)])
                            if bk is None or k<bk: bk=k;bestm=a
                        p=bestm
                preds.append(int(p) if p is not None else 2)
                continue
            else:
                p=_P5preds.get(rid)
                if p is None:
                    REASON['multileg_None']+=1
                    # fallback greedy Manhattan+crowded ve target gan nhat
                    _lms={}
                    for rc,t in rc2lm.items(): _lms.setdefault(t,[]).append(rc)
                    _tl=_lms.get(mission['via'] or mission['goal']) or [rc for v in _lms.values() for rc in v]
                    if _tl:
                        bestm=None;bk=None
                        for a,nxt in [(0,(robot_rc[0]-1,robot_rc[1])),(1,(robot_rc[0]+1,robot_rc[1])),(2,(robot_rc[0],robot_rc[1]-1)),(3,(robot_rc[0],robot_rc[1]+1))]:
                            if nxt not in rcset: continue
                            found=next((e for e in edge_list if (tuple(e['a'])==robot_rc and tuple(e['b'])==nxt) or (tuple(e['b'])==robot_rc and tuple(e['a'])==nxt)), None)
                            if found and (found['status']=='closed' or found['stairs']): continue
                            ec=(1 if (found and found.get('status')=='crowded') else 0)
                            d=min(abs(nxt[0]-t[0])+abs(nxt[1]-t[1]) for t in _tl); k=(d+ec,REL[(robot_hd,a)])
                            if bk is None or k<bk: bk=k;bestm=a
                        p=bestm
                preds.append(int(p) if p is not None else 2)
        if (si+1)%200==0: print(f'done {si+1}/{len(test_pred)}',flush=True)
pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/predictions_v5_test.json').write_text(json.dumps(preds))
print(f'saved val-cv {len(preds)}')
# score macro vs val labels
print('REASONS', dict(REASON))
