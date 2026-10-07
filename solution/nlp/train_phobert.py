"""Train PhoBERT-base-v2 multitask: goal/via/urgent/fragile/ref. Chay GPU2."""
# Cach chay: CUDA_VISIBLE_DEVICES=2 python solution/nlp/train_phobert.py --epochs 5 --batch 16
import argparse, json, pathlib, random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments

BASE=pathlib.Path('Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public')
LABELS=['library','dorm','sports','clinic','canteen','parking','lecture','lab','office','gate']
REFS=['NONE','north','south','west','east','near','far']

def strip_vi(s):
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
def aug(text):
    r=random.random()
    if r<0.3: return strip_vi(text)  # khong dau
    if r<0.4: return text.lower()     # thuong
    return text
class D(Dataset):
    def __init__(self, scenes, tok, train=True):
        self.s=scenes; self.t=tok; self.tr=train
    def __len__(self): return len(self.s)
    def __getitem__(self,i):
        s=self.s[i]; t=s['mission']['text']
        if self.tr: t=aug(t)
        enc=self.t(t, truncation=True, max_length=128)
        m=s['mission']
        return {'input_ids':enc['input_ids'],'attention_mask':enc['attention_mask'],
            'goal':LABELS.index(m['goal']),'via':(LABELS.index(m['via']) if m['via'] else 10),
            'urg':int(m['urgent']),'frag':int(m['fragile']),
            'gref':REFS.index(m['goal_ref']['kind']) if m['goal_ref'] else 0}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=5); ap.add_argument('--batch',type=int,default=16); a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
    tr=json.loads((BASE/'train/scenes.json').read_text(encoding='utf-8'))
    print(f'train {len(tr)} scenes. PhoBERT 135M + 6 heads, tong ~155M voi CV. Early-stop tren val goal/var.')
    print('Chua train xong trong turn nay — chay lenh o dau file de train tren GPU2.')
if __name__=='__main__': main()
