"""Shared NLP rules: gref_kind (rule_kind4) + anchor (alias+single+spatial+negation). Train-derived only."""
import unicodedata, re, pickle, pathlib
from collections import Counter
def strip(s): return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn').lower()
NORTH=['phia tren','ben tren','man tren','tren cung','phia bac','man bac','mep tren','ria tren','cao nhat','ve phia bac','phia tren ban do','man bac','o tren','tren ban do','dinh ban do','cuc tren','goc tren','sat mep tren']
SOUTH=['phia duoi','ben duoi','man duoi','duoi cung','phia nam','man nam','mep duoi','ria duoi','ve phia nam','phia duoi ban do','o duoi','duoi ban do','cuc duoi','goc duoi','sat mep duoi','thap nhat']
EAST=['ben phai','phia phai','man phai','mep phai','ria phai','phia dong','man dong','ve phia dong','o phai','ngoai cung ben phai','sat mep phai','cuc phai','goc phai','phia phai ban do']
WEST=['ben trai','phia trai','man trai','mep trai','ria trai','phia tay','man tay','ve phia tay','o trai','ngoai cung ben trai','sat mep trai','cuc trai','goc trai','phia trai ban do']
def _has(t,pats): return any(p in t for p in pats)
def rule_kind(text):
    t=' '+strip(text)+' '
    has_nhat='nhat' in t
    has_cung=re.search(r'\bcung\b',t) is not None
    has_hon=re.search(r'\bhon\b',t) is not None
    has_gan=re.search(r'\bgan\b',t) is not None
    has_xa=re.search(r'\bxa\b',t) is not None
    n=_has(t,NORTH); s=_has(t,SOUTH); e=_has(t,EAST); w=_has(t,WEST)
    ndir=sum([n,s,e,w])
    sup=(has_nhat or has_cung)
    if sup and ndir==1:
        if n: return 'north_most'
        if s: return 'south_most'
        if e: return 'east_most'
        if w: return 'west_most'
    if has_nhat and ndir==0:
        return 'anchor_near'
    if sup and ndir>1: return None
    if not sup:
        if has_hon and has_xa and not has_gan: return 'far'
        if has_hon and has_gan and not has_xa: return 'near'
        if has_hon and has_xa and has_gan: return 'far' if t.find('xa')<t.find('gan') else 'near'
        if ndir==1:
            if n: return 'north'
            if s: return 'south'
            if e: return 'east'
            if w: return 'west'
        if ndir==0 and not has_hon and not has_gan and not has_xa: return 'NONE'
    return None
# aliases (manual + train-mined)
_ALIAS_PKL='/tmp/opencode/mined_alias.pkl'
_MANUAL={
 'library':['thu vien','phong doc','tra sach','muon sach','giao trinh','thu thu','khu doc sach','phong tai lieu','tai lieu'],
 'dorm':['ky tuc xa','ktx','khu noi tru','nha o sinh vien','noi tru','day phong noi tru','phong o sinh vien','toa nha o'],
 'sports':['nha thi dau','san tap','san bong','nha the chat','san the thao','san luyen tap','the chat'],
 'clinic':['tram y te','phong kham','tram y','noi kham','phong so cuu','so cuu','tram xa','khu y te'],
 'canteen':['can tin','cang tin','nha an','bep an','khu an uong','phong an','bep truong'],
 'parking':['bai do xe','khu gui xe','nha xe','bai xe','cho do xe','ham xe','cho de xe','noi giu xe','de xe','giu xe','khu dau xe'],
 'lecture':['giang duong','hoi truong','lop hoc','phong hoc','khu giang duong','phong hoc lon','toa giang duong'],
 'lab':['phong thi nghiem','phong thuc nghiem','xuong thuc hanh','lab hoa','phong lab','thi nghiem','khu thi nghiem'],
 'office':['phong hanh chinh','van phong','hieu bo','toa hieu bo','phong giao vu','noi nop ho so','khu hanh chinh','phong mot cua'],
 'gate':['cong truong','cong vao','loi vao','cong chinh','chot cong','cong truong'],
}
def _load_alias():
    d={}
    try: mined=pickle.load(open(_ALIAS_PKL,'rb'))
    except: mined={}
    for k in _MANUAL:
        dd={}
        for a in _MANUAL[k]: dd[strip(a)]=10.0
        for ph,cc,r in mined.get(k,[]):
            ph2=re.sub(r'\s+(nhat|hon|cung)$','',ph)
            if len(ph2)<5 or r<4: continue
            if len(ph2.split())==1 and r<8: continue
            dd[ph2]=max(dd.get(ph2,0),float(r))
        for bad in ['sat','gan','canh','nam sat','nam gan','nam canh','noi canh','diem gan','diem sat']:
            dd.pop(bad,None)
        d[k]=dd
    return d
ALIAS=_load_alias()
_SPAT=re.compile(r'(gan|xa|canh|sat|ke|nhat|hon|cung|mep|ria)\b')
_NEG=re.compile(r'(khong|dung|bo qua|huy|khong can|dung nham|khong phai)')
def match_anchor(text, goal, via, landmarks):
    """landmarks: list of {type, rc} or [(type, rc)]. Returns anchor type or None."""
    t=strip(text.lower())
    if landmarks and isinstance(landmarks[0],dict):
        cnt=Counter(lm['type'] for lm in landmarks)
    else:
        cnt=Counter(lm[0] for lm in landmarks)
    spat=[m.start() for m in _SPAT.finditer(t)]
    negs=[m.start() for m in _NEG.finditer(t)]
    scored=[]
    for k in cnt:
        if cnt[k]!=1: continue
        wtype=0.5 if k in (goal,via) else 1.0
        tot=0; prox=False
        for a,w in ALIAS.get(k,{}).items():
            idx=t.find(a)
            if idx<0: continue
            neg=any(0<=(idx-p)<=15 for p in negs)
            ww=w*(0.25 if neg else 1.0)
            d=min([abs(idx-p) for p in spat],default=60)
            if d<=35: prox=True
            tot+=ww*max(0.3,(1-d/60))*wtype
        if tot>0 and prox: scored.append((-tot,k))
    if not scored: return None
    scored.sort()
    return scored[0][1]
