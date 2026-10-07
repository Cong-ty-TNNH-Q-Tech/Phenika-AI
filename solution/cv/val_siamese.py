"""Validate Siamese mean-RGB vs absolute CNN tren val (true legend boxes + true edges)."""
import json, pathlib
from PIL import Image
import numpy as np
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
scenes=json.loads((BASE/'validation/scenes.json').read_text())[:50]
def mean_rgb(im):
    a=np.array(im.resize((32,32))).reshape(-1,3).mean(0)
    return a
tot=hit_siam=0; swap_tot=hit_swap=0; noswap_tot=hit_noswap=0
for s in scenes:
    im=Image.open(BASE/'validation'/s['image']).convert('RGB')
    # legend swatches true boxes
    sw={}
    for lg in s['legend']:
        if lg['kind'] in ('normal','crowded','covered','closed'):
            x0,y0,x1,y1=[int(v) for v in lg['swatch']]
            crop=im.crop((x0,y0,x1,y1))
            sw[lg['kind']]=mean_rgb(crop)
    xy={tuple(n['rc']):n['xy'] for n in s['nodes']}
    swapped=not (s['road_look']['normal']=='normal' and s['road_look']['crowded']=='crowded' and s['road_look']['covered']=='covered')
    for e in s['edges']:
        a,b=xy[tuple(e['a'])],xy[tuple(e['b'])]
        mx,my=(a[0]+b[0])/2,(a[1]+b[1])/2
        crop=im.crop((max(0,int(mx-32)),max(0,int(my-32)),int(mx+32),int(my+32)))
        em=mean_rgb(crop)
        bestk=min(sw, key=lambda k:float(((em-sw[k])**2).sum()))
        tot+=1
        if bestk==e['status']: hit_siam+=1
        if swapped: swap_tot+=1; hit_swap+=(bestk==e['status'])
        else: noswap_tot+=1; hit_noswap+=(bestk==e['status'])
print(f'Siamese meanRGB val50: overall {hit_siam}/{tot}={hit_siam/max(1,tot):.4f}, noswap {hit_noswap}/{noswap_tot}={hit_noswap/max(1,noswap_tot):.4f}, swap {hit_swap}/{swap_tot}={hit_swap/max(1,swap_tot):.4f}')
print('So voi CNN tuyet doi 59% (val400 crops). Siamese can ~85%+ moi dat.')
