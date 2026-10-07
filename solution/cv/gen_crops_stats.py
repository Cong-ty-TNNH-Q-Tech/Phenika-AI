"""Gen CV crops: edge status + stats. Chay CPU, nhanh."""
import json, pathlib
from PIL import Image
from collections import Counter
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
for split in ['train','validation']:
    scenes=json.loads((BASE/split/'scenes.json').read_text())
    c_status=Counter(); c_stairs=Counter(); c_oneway=Counter()
    n_edges=0
    for s in scenes:
        for e in s['edges']:
            c_status[e['status']]+=1; c_stairs[e['stairs']]+=1; c_oneway[e['oneway_to'] is not None]+=1; n_edges+=1
    print(f'{split}: scenes={len(scenes)} edges={n_edges} status={dict(c_status)} stairs={dict(c_stairs)} oneway={dict(c_oneway)}')
    # vi du 1 scene: kich thuoc crop edge
    s=scenes[0]
    im=Image.open(BASE/split/s['image']).convert('RGB')
    print(f'  ex image {s["image"]} size={im.size} grid={s["grid"]} nodes={len(s["nodes"])} style={s["style"]} road_look={s["road_look"]}')
    xy={tuple(n['rc']):n['xy'] for n in s['nodes']}
    e=s['edges'][0]
    a,b=xy[tuple(e['a'])],xy[tuple(e['b'])]
    print(f'  ex edge {e["a"]}->{e["b"]} status={e["status"]} a_xy={a} b_xy={b}')
print('OK gen stats. Tiep: crop edge 64x64 tai midpoint + legend swatch de train Siamese.')
