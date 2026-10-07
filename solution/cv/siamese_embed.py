"""Embedding Siamese: MobileNet features cosine vs meanRGB, val50."""
import json, pathlib
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as NF
from torchvision import models, transforms
import numpy as np
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
TF=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# backbone tu edge_mobilenet (da train status) -> features truoc classifier
m=models.mobilenet_v3_small(weights=None)
m.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/edge_mobilenet.pt',map_location=dev),strict=False)
# lay features: dung avgpool output? mobilenet_v3_small: features -> avgpool -> classifier. Lay features+avgpool flatten.
m=m.to(dev).eval()
def embed(im):
    with torch.no_grad():
        x=TF(im).unsqueeze(0).to(dev)
        f=m.features(x)
        f=m.avgpool(f)
        return f.flatten().cpu()
def mean_rgb(im):
    return np.array(im.resize((32,32))).reshape(-1,3).mean(0)
scenes=json.loads((BASE/'validation/scenes.json').read_text())[:30]
tot=hit_mean=hit_emb=0
for s in scenes:
    im=Image.open(BASE/'validation'/s['image']).convert('RGB')
    sw_img={}; sw_emb={}
    for lg in s['legend']:
        if lg['kind'] in ('normal','crowded','covered','closed'):
            x0,y0,x1,y1=[int(v) for v in lg['swatch']]
            crop=im.crop((x0,y0,x1,y1))
            sw_img[lg['kind']]=mean_rgb(crop)
            sw_emb[lg['kind']]=embed(crop)
    xy={tuple(n['rc']):n['xy'] for n in s['nodes']}
    for e in s['edges'][:40]:
        a,b=xy[tuple(e['a'])],xy[tuple(e['b'])]
        mx,my=(a[0]+b[0])/2,(a[1]+b[1])/2
        crop=im.crop((max(0,int(mx-32)),max(0,int(my-32)),int(mx+32),int(my+32)))
        em=mean_rgb(crop); ee=embed(crop)
        bm=min(sw_img,key=lambda k:float(((em-sw_img[k])**2).sum()))
        be=min(sw_emb,key=lambda k:float((1-NF.cosine_similarity(ee,sw_emb[k],dim=0)).item()))
        tot+=1; hit_mean+=(bm==e['status']); hit_emb+=(be==e['status'])
print(f'val30 sample: meanRGB {hit_mean}/{tot}={hit_mean/max(1,tot):.4f}, embedding-cosine {hit_emb}/{tot}={hit_emb/max(1,tot):.4f}')
