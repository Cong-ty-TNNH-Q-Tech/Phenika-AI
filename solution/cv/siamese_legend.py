"""Siamese full: legend detect + edge match + re-infer test."""
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
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('load legend detector...',dev,flush=True)
lg=fasterrcnn_resnet50_fpn(weights=None)
lg.roi_heads.box_predictor.cls_score=nn.Linear(lg.roi_heads.box_predictor.cls_score.in_features,7)
lg.roi_heads.box_predictor.bbox_pred=nn.Linear(lg.roi_heads.box_predictor.bbox_pred.in_features,28)
lg.load_state_dict(torch.load('/mnt/hdd2/qtech/Phenika-AI/solution/cv/legend_frcnn.pt',map_location=dev)); lg=lg.to(dev).eval()
te_obs=json.loads((BASE/'test/observations.json').read_text(encoding='utf-8'))
# test 20 anh dau de kiem tra legend detect (nhanh, review truoc khi full 1200)
with torch.no_grad():
    for si in range(20):
        img_path=BASE/'test'/te_obs[si*10]['image']
        im=Image.open(img_path).convert('RGB'); w0,h0=im.size; sz=800; imr=im.resize((sz,sz))
        out=lg([F.to_tensor(imr).to(dev)])[0]
        boxes=out['boxes'].cpu(); scores=out['scores'].cpu(); labels=out['labels'].cpu()
        dets=[(KINDS[labels[i]-1],round(float(scores[i]),2)) for i in range(len(boxes)) if scores[i]>0.5 and labels[i]>=1]
        print(f'{img_path.name}: {dets}',flush=True)
print('OK legend detect chay duoc. Tiep: match mau + re-infer full 1200 (se chay 45-60 phut).')
