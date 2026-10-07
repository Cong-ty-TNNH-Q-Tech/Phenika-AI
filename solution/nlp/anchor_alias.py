"""TTA NLP + near/far anchor bang alias tu DE_BAI/train (khong doc test de lam luat)."""
import json, pathlib, unicodedata, re
BASE=pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
def strip(s): return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
ALIASES={
 'library':['thu vien','tv','giao trinh','thu thu','sach'],
 'dorm':['ky tuc','ktx','noi tru','sinh vien','giang duong'],  # se loc bot gay nhieu
 'sports':['the thao','tt','san tap','san bong','bong'],
 'clinic':['tram y','yt','benh','kham'],
 'canteen':['can tin','ca','an uong','bep','com'],
 'parking':['bai xe','xe','giu xe','do xe','gui xe'],
 'lecture':['giang duong','gd','hoi truong','lop','giang'],
 'lab':['lab','tn','thi nghiem','thuc nghiem','phong lab'],
 'office':['hanh chinh','hc','van phong','khoa'],
 'gate':['cong','ct','cong truong'],
}
# Chuan hoa alias (strip, lower)
for k in ALIASES: ALIASES[k]=[strip(a.lower()) for a in ALIASES[k]]
print('alias ok')
# test anchor demo: lay 5 mission near/far du doan
test_pred=json.loads(pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/test_missions_pred.json').read_text(encoding='utf-8'))
test_gref=json.loads(pathlib.Path('/mnt/hdd2/qtech/Phenika-AI/solution/nlp/test_gref_pred.json').read_text(encoding='utf-8'))
cnt=0
for m,g in zip(test_pred,test_gref):
    if g in ('near','far'):
        cnt+=1
        if cnt<=5:
            t=strip(m['text'].lower())
            found=[k for k,als in ALIASES.items() for a in als if a in t]
            print(g, m['goal'], found, m['text'][:80])
        if cnt>200: break
print('near/far count:',cnt)
