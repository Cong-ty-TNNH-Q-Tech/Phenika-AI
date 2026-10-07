"""Anchor rule v2: alias mo rong (mine tu train) + pattern gan/xa + single-instance."""
import json, pathlib, unicodedata, re
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
def strip(s): return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
ALIASES={
 'library':['thu vien','noi muon giao trinh','cho tra sach','phong doc','tra sach','muon sach','khu doc sach','noi muon sach tham khao','ham xe cach xa thu vien','xa thu vien'],
 'dorm':['ky tuc xa','ktx','khu noi tru','toa nha o cua sinh vien','phong o sinh vien','nha o sinh vien','day phong noi tru','khu o cua sinh vien','phong cong tac sinh vien'],
 'sports':['nha thi dau','san tap','san bong','doi bong','nha the chat','san luyen tap','san the thao','khu the thao'],
 'clinic':['tram y te','tram xa','phong kham','phong y te','tram y','noi kham suc khoe','tram y te lay bo bang gac'],
 'canteen':['can tin','cang tin','nha an','bep an','bep truong','khu an uong','phong an tap the','noi phuc vu bua trua','sat can tin','xa nha an'],
 'parking':['bai do xe','khu gui xe','nha xe','bai xe','cho do xe','khu dau xe','ham xe','khu gui xe hon','bai do xe xong'],
 'lecture':['phong hoc lon','khu giang duong','hoi truong hoc','lop hoc','giang duong','phong hoc','day phong hoc','noi len lop','phong giao vu','khu hanh chinh nam gan cong'],
 'lab':['phong lab','phong thi nghiem','xuong thuc hanh','lab hoa','phong thuc nghiem','phong thuc hanh','san luyen tap qua'],
 'office':['phong mot cua','toa hieu bo','khu hanh chinh','van phong khoa','hieu bo','phong giao vu','noi nop ho so','phong dao tao'],
 'gate':['cong truong','cong vao','loi vao truong','cong chinh','cong a','chot bao ve cong','chot cong','loi vao'],
}
for k in ALIASES: ALIASES[k]=sorted(set(strip(a.lower()) for a in ALIASES[k]),key=len,reverse=True)
def predict_anchor(text, goal, via, landmarks):
    from collections import Counter
    t=strip(text.lower())
    cnt=Counter(lm['type'] for lm in landmarks)
    cands=[]
    # tim pattern gan/xa/sat/cach xa + alias trong 40 ky tu sau
    for m in re.finditer(r'(gan|xa|sat|cach xa|sat |man duoi|man tren)\b',t):
        window=t[m.end():m.end()+40]
        for k,als in ALIASES.items():
            if k in (goal,via): continue
            for a in als:
                if a in window and cnt.get(k,0)>=1:
                    # loai phu dinh: neu co 'khong' trong 10 ky tu truoc alias? don gian bo qua
                    cands.append(k); break
    if cands:
        # chon single-instance truoc, else dau tien
        for k in cands:
            if cnt.get(k,0)==1: return k
        return cands[0]
    # fallback: alias xuat hien (khong phu dinh) + single
    found=[]
    for k,als in ALIASES.items():
        if k in (goal,via): continue
        for a in als:
            idx=t.find(a)
            if idx>=0:
                # loai neu co khong/dung trong 12 ky tu truoc
                pre=t[max(0,idx-12):idx]
                if any(n in pre for n in ['khong','dung','bo qua']): continue
                found.append(k); break
    for k in found:
        if cnt.get(k,0)==1: return k
    return found[0] if found else None
for split in ['train','validation']:
    scenes=json.loads((BASE/split/'scenes.json').read_text())
    tot=hit=0
    for s in scenes:
        m=s['mission']
        true=None
        for r in [m['goal_ref'],m['via_ref']]:
            if r and r.get('kind') in ('near','far'): true=r.get('anchor')
        if not true: continue
        tot+=1
        pred=predict_anchor(m['text'],m['goal'],m['via'],s['landmarks'])
        if pred==true: hit+=1
    print(f'{split} anchor rule v2: {hit}/{tot}={hit/max(1,tot):.4f}')
