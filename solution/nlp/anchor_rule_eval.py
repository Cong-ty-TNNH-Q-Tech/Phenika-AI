"""Anchor rule: alias (tu train, khong test) + single-instance + loai distractor don gian."""
import json, pathlib, unicodedata, re, collections
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
def strip(s): return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
# alias rut gon, tranh overlap (bo 'xe','giang duong' chung chung)
ALIASES={
 'library':['thu vien','giao trinh','thu thu'],
 'dorm':['ky tuc xa','ktx','khu noi tru'],
 'sports':['nha thi dau','san tap','san bong','doi bong'],
 'clinic':['tram y te','tram xa','phong kham'],
 'canteen':['can tin','khu an uong','bep truong'],
 'parking':['bai giu xe','khu gui xe','cho do xe'],
 'lecture':['giang duong','hoi truong'],
 'lab':['phong lab','phong thi nghiem','lab'],
 'office':['van phong khoa','phong hanh chinh'],
 'gate':['cong truong','cong chinh'],
}
for k in ALIASES: ALIASES[k]=[strip(a.lower()) for a in ALIASES[k]]
scenes=json.loads((BASE/'train/scenes.json').read_text())
tot=hit=0
for s in scenes:
    m=s['mission']
    true_anchor=None
    for r in [m['goal_ref'],m['via_ref']]:
        if r and r.get('kind') in ('near','far'): true_anchor=r.get('anchor')
    if not true_anchor: continue
    tot+=1
    t=strip(m['text'].lower())
    # loai cau phu dinh don gian: cat bo phan sau 'khong can','dung nham','khong phai','bo qua'
    for pat in ['khong can','dung nham','khong phai','bo qua','khong phai o']:
        idx=t.find(pat)
        if idx>=0: t=t[:idx]  # chi giu phan truoc distractor (don gian, co the sai khi anchor sau distractor - chap nhan)
    found=[]
    for k,als in ALIASES.items():
        for a in als:
            if a in t and k not in (m['goal'],m['via']):
                found.append(k); break
    # single-instance constraint (true landmarks)
    from collections import Counter
    cnt=Counter(lm['type'] for lm in s['landmarks'])
    singles=[k for k in found if cnt.get(k,0)==1]
    pred=singles[0] if singles else (found[0] if found else None)
    if pred==true_anchor: hit+=1
print(f'anchor rule train near/far: {hit}/{tot}={hit/max(1,tot):.4f}')
