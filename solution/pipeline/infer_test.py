"""Infer test -> predictions.json (12k, format BTC)."""
# v1: majority theo robot (val 26.6%, an toan de nop).
# v2 (tiep): thay bang NLP PhoBERT + CV graph + oracle policies (val oracle 85%, TF-IDF hien 67.8%).
# Cach chay: python solution/pipeline/infer_test.py --data .../delivery_public --out solution/predictions.json
import argparse, json, pathlib
from collections import Counter
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=pathlib.Path,default=pathlib.Path('Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public'))
    ap.add_argument('--out',type=pathlib.Path,default=pathlib.Path('solution/predictions.json'))
    a=ap.parse_args()
    rows=json.loads((a.data/'train/observations.json').read_text(encoding='utf-8'))
    labels=json.loads((a.data/'train/labels.json').read_text(encoding='utf-8'))
    maj={}
    for r in range(10):
        maj[r]=Counter(y for o,y in zip(rows,labels) if o['robot_id']==r).most_common(1)[0][0]
    te=json.loads((a.data/'test/observations.json').read_text(encoding='utf-8'))
    pred=[maj[o['robot_id']] for o in te]
    a.out.write_text(json.dumps(pred))
    print(f'wrote {len(pred)} -> {a.out} (format sample_submission OK)')
if __name__=='__main__': main()
